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
    """
    A diffusion model for super-resolution image generation.
    """

    def __init__(self, opt):
        """
        Initialize the DiffusionModel.

        Args:
            opt: Configuration options for the model.
        """
        super().__init__()
        self.opt = opt
        self.normalizer = layers.Normalization()
        self.network = get_network(self.opt)
        self.ema_network = keras.models.clone_model(self.network)

    def compile(self, **kwargs):
        """
        Compile the model and set up metrics.
        """
        super().compile(**kwargs)
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(self.opt, name="kid")

    @property
    def metrics(self):
        """
        Define the metrics for the model.
        """
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        """
        Convert the pixel values back to 0-1 range.

        Args:
            images: Normalized images.

        Returns:
            Denormalized images.
        """
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return ops.clip(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        """
        Calculate the noise and signal rates for given diffusion times.

        Args:
            diffusion_times: Tensor of diffusion times.

        Returns:
            Tuple of noise_rates and signal_rates.
        """
        start_angle = ops.cast(ops.arccos(self.opt.max_signal_rate), "float32")
        end_angle = ops.cast(ops.arccos(self.opt.min_signal_rate), "float32")
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = ops.cos(diffusion_angles)
        noise_rates = ops.sin(diffusion_angles)
        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        """
        Denoise the input images.

        Args:
            noisy_images: Input noisy images.
            noise_rates: Noise rates for the images.
            signal_rates: Signal rates for the images.
            training: Boolean indicating whether in training mode.

        Returns:
            Tuple of predicted noises and predicted images.
        """
        network = self.network if training else self.ema_network
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images[:,:,:,3:] - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, lr_images, diffusion_steps):
        """
        Perform reverse diffusion (sampling).

        Args:
            initial_noise: Initial noise for the diffusion process.
            lr_images: Low-resolution input images.
            diffusion_steps: Number of diffusion steps to perform.

        Returns:
            Generated high-resolution images.
        """
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        next_noisy_images = initial_noise

        for step in range(diffusion_steps):
            noisy_images = next_noisy_images
            noisy_images = keras.layers.Concatenate(axis=3)([lr_images, noisy_images])

            diffusion_times = ops.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=False)

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = next_signal_rates * pred_images + next_noise_rates * pred_noises

        return pred_images

    def train_step(self, images):
        """
        Perform a single training step.

        Args:
            images: Tuple of high-resolution and low-resolution images.

        Returns:
            Dict of metric results.
        """
        images, lr_images = images
        lr_images = self._preprocess_images(lr_images)
        images = self.normalizer(images, training=True)

        noises = keras.random.normal(shape=(self.opt.batch_size, self.opt.image_size, self.opt.image_size, 3))
        diffusion_times = keras.random.uniform(shape=(self.opt.batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        noisy_images = keras.layers.Concatenate(axis=3)([lr_images, noisy_images])

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
            noise_loss = self.loss(noises, pred_noises)
            image_loss = self.loss(images, pred_images)

        self._update_model(tape, noise_loss)
        self._update_metrics(noise_loss, image_loss)
        self._update_ema()

        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        """
        Perform a single test step.

        Args:
            images: Tuple of high-resolution and low-resolution images.

        Returns:
            Dict of metric results.
        """
        images, lr_images = images
        lr_images = self._preprocess_images(lr_images)
        images = self.normalizer(images, training=False)

        noises = keras.random.normal(shape=(self.opt.batch_size, self.opt.image_size, self.opt.image_size, 3))
        diffusion_times = keras.random.uniform(shape=(self.opt.batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        noisy_images = keras.layers.Concatenate(axis=3)([lr_images, noisy_images])

        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self._update_metrics(noise_loss, image_loss)
        self._compute_kid(images, lr_images, noises)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, images, plot_diffusion_steps=20, epoch=999, logs=None, num_rows=1, num_cols=8):
        """
        Plot and save generated, target, and input images.

        Args:
            images: Tuple of high-resolution and low-resolution images.
            plot_diffusion_steps: Number of diffusion steps for plotting.
            epoch: Current epoch number.
            logs: Training logs (not used).
            num_rows: Number of rows in the plot.
            num_cols: Number of columns in the plot.
        """
        images, lr_images = images
        lr_images = self._preprocess_images(lr_images)
        noises = keras.random.normal(shape=(self.opt.batch_size, self.opt.image_size, self.opt.image_size, 3))

        generated_images = self.reverse_diffusion(noises, lr_images, plot_diffusion_steps)
        generated_images = self.denormalize(generated_images)
        lr_images = self.denormalize(lr_images)

        self._save_image_grid(generated_images, epoch, "out", num_rows, num_cols)
        self._save_image_grid(images, epoch, "tar", num_rows, num_cols)
        self._save_image_grid(lr_images, epoch, "inp", num_rows, num_cols)

    def _preprocess_images(self, images):
        """Preprocess input images."""
        return self.normalizer(keras.layers.Resizing(self.opt.image_size, self.opt.image_size, interpolation="nearest")(images), training=False)

    def _update_model(self, tape, loss):
        """Update model weights."""
        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

    def _update_metrics(self, noise_loss, image_loss):
        """Update model metrics."""
        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

    def _update_ema(self):
        """Update Exponential Moving Average of the model weights."""
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.opt.ema * ema_weight + (1 - self.opt.ema) * weight)

    def _compute_kid(self, images, lr_images, noises):
        """Compute Kernel Inception Distance (KID) metric."""
        images = self.denormalize(images)
        generated_images = self.reverse_diffusion(noises, lr_images, self.opt.kid_diffusion_steps)
        generated_images = self.denormalize(generated_images)
        self.kid.update_state(images, generated_images)

    def _save_image_grid(self, images, epoch, suffix, num_rows, num_cols):
        """Save a grid of images."""
        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for i in range(num_rows * num_cols):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(images[i])
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(f'{self.opt.data_name}/val_images/{epoch}_{suffix}.png', bbox_inches='tight')
        plt.close()