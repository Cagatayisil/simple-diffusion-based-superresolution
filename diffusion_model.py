

import math
import keras
from keras import layers
from keras import ops
from network_modules import *
from utils import KID
import tensorflow as tf
import matplotlib.pyplot as plt




@keras.saving.register_keras_serializable()
class DiffusionModel(keras.Model):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # image_size = opt.image_size
        # widths = opt.widths
        # block_depth = opt.block_depth

        self.normalizer = layers.Normalization()
        self.network = get_network(self.opt)
        self.ema_network = keras.models.clone_model(self.network)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(self.opt,name="kid")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return ops.clip(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = ops.cast(ops.arccos(self.opt.max_signal_rate), "float32")
        end_angle = ops.cast(ops.arccos(self.opt.min_signal_rate), "float32")

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = ops.cos(diffusion_angles)
        noise_rates = ops.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images[:,:,:,3:] - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, lr_images, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            noisy_images = keras.layers.Concatenate(axis=3)([lr_images, noisy_images])###

            # separate the current noisy image to its components
            diffusion_times = ops.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    # def generate(self, num_images, diffusion_steps):
    #     # noise -> images -> denormalized images
    #     initial_noise = keras.random.normal(
    #         shape=(num_images, image_size, image_size, 3)
    #     )
    #     generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
    #     generated_images = self.denormalize(generated_images)
    #     return generated_images

    def train_step(self, images):
        image_size = self.opt.image_size
        batch_size = self.opt.batch_size
        ema = self.opt.ema


        down_sz = int(image_size/4)
        lr_images = keras.layers.Resizing(down_sz,down_sz, interpolation="nearest")(images)
        lr_images = keras.layers.Resizing(image_size,image_size, interpolation="nearest")(lr_images)
        # lr_images = images
        lr_images = self.normalizer(lr_images, training=True)
        # normalize images to have standard deviation of 1, like the noises
        # print(images.shape)
        images = self.normalizer(images, training=True)
        # print(images.shape)
        # asdasd = keras.random.normal(shape=())
        # print(lr_images.shape)
        # print(noises.shape)
        noises = keras.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        noisy_images = keras.layers.Concatenate(axis=3)([lr_images, noisy_images])

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):

        image_size = self.opt.image_size
        batch_size = self.opt.batch_size
        kid_diffusion_steps = self.opt.kid_diffusion_steps


        down_sz = int(image_size/4)
        lr_images = keras.layers.Resizing(down_sz,down_sz, interpolation="nearest")(images)
        lr_images = keras.layers.Resizing(image_size,image_size, interpolation="nearest")(lr_images)

        lr_images = self.normalizer(lr_images, training=False)

        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = keras.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        noisy_images = keras.layers.Concatenate(axis=3)([lr_images, noisy_images])

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        # generated_images = self.generate(
        #     num_images=batch_size, diffusion_steps=kid_diffusion_steps
        # )

        initial_noise = noises
        generated_images = self.reverse_diffusion(initial_noise, lr_images, kid_diffusion_steps)
        generated_images = self.denormalize(generated_images)


        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, images, plot_diffusion_steps = 20, epoch=999, logs=None, num_rows=1, num_cols=8):

        image_size = self.opt.image_size
        batch_size = self.opt.batch_size
        image_save_path = f'{self.opt.data_name}/val_images'

        down_sz = int(image_size/4)
        lr_images = keras.layers.Resizing(down_sz,down_sz, interpolation="nearest")(images)
        lr_images = keras.layers.Resizing(image_size,image_size, interpolation="nearest")(lr_images)
        lr_images = self.normalizer(lr_images, training=False)
        noises = keras.random.normal(shape=(batch_size, image_size, image_size, 3))

        generated_images = self.reverse_diffusion(noises, lr_images, plot_diffusion_steps)
        generated_images = self.denormalize(generated_images)
        lr_images = self.denormalize(lr_images)

        # plot random generated images for visual evaluation of generation quality
        # generated_images = self.generate(
        #     num_images=num_rows * num_cols,
        #     diffusion_steps=plot_diffusion_steps,
        # )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        plt.savefig(f'{image_save_path}/{epoch}_out.png', bbox_inches='tight')
        # plt.show()
        plt.close()

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(images[index])
                plt.axis("off")
        plt.tight_layout()
        plt.savefig(f'{image_save_path}/{epoch}_tar.png', bbox_inches='tight')
        # plt.show()
        plt.close()

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(lr_images[index])
                plt.axis("off")
        plt.tight_layout()
        plt.savefig(f'{image_save_path}/{epoch}_inp.png', bbox_inches='tight')
        # plt.show()
        plt.close()