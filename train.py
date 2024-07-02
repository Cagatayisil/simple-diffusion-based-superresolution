import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

import keras
from keras import layers
from keras import ops
import argparse
from diffusion_model import DiffusionModel


def init_parameters():
    parser = argparse.ArgumentParser(description='superresolution')
    parser.add_argument('--gpu_id', type=int, default=0, help='the ID of the visible GPU device (only used when not in parallel mode)')

    parser.add_argument('--data_name', type=str, default='SR4_2', help='')

    parser.add_argument('--num_epochs', type=int, default=100, help='epoch')
    parser.add_argument('--image_size', type=int, default=64, help='img size for data loading')

    parser.add_argument('--kid_image_size', type=int, default=75, help='kid_image_size')
    parser.add_argument('--kid_diffusion_steps', type=int, default=5, help='kid_diffusion_steps')
    parser.add_argument('--plot_diffusion_steps', type=int, default=20, help='plot_diffusion_steps')
    parser.add_argument('--min_signal_rate', type=float, default=0.02, help='sampling')
    parser.add_argument('--max_signal_rate', type=float, default=0.95, help='sampling')

# architecture
    parser.add_argument('--embedding_dims', type=int, default=32, help='embedding_dims')
    parser.add_argument('--embedding_max_frequency', type=float, default=1000.0, help='embedding_max_frequency')
    parser.add_argument('--block_depth', type=int, default=2, help='block_depth')

# optimization
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for data loading')
    parser.add_argument('--ema', type=float, default=0.999, help='ema')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning_rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')

    vc = parser.parse_args()
    # vc.record_file = vc.train_path

    vc.widths = [32, 64, 96, 128]

    if not os.path.exists(f'{vc.data_name}'):
        os.makedirs(f'{vc.data_name}')

    vc.config_file = f'{vc.data_name}/config_file.txt' # training config file
    cfg = ''
    cfg += '----------------- Options ---------------\n'
    for k, v in sorted(vars(vc).items()):
        cfg += f'{str(k):>25}: {str(v):<30}\n'
    cfg += '----------------- End -------------------\n'
    # cfg += '----------------- Options ---------------\n'
    # for k, v in sorted(vars(vc).items()):
    #     cfg += f'{str(k):>25}: {str(v):<30}\n'
    # cfg += '----------------- End -------------------'
    print(cfg)
    with open(vc.config_file, 'w') as f:
        f.writelines(f'{cfg}')


    return vc


if __name__ == '__main__':
    train_config = init_parameters()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{train_config.gpu_id}"


    if not os.path.exists(f'{train_config.data_name}/val_images/'):
        os.makedirs(f'{train_config.data_name}/val_images/')
    if not os.path.exists(f'{train_config.data_name}/Models/last/'):
        os.makedirs(f'{train_config.data_name}/Models/last/')
    # if not os.path.exists(f'{valid_config.data_name}/Models/best/'):
    #         os.makedirs(f'{valid_config.data_name}/Models/best/')

    image_size = train_config.image_size



    dataset_name = "oxford_flowers102"
    dataset_repetitions = 5
    ####
    def preprocess_image(data):
        # center crop image
        height = ops.shape(data["image"])[0]
        width = ops.shape(data["image"])[1]
        crop_size = ops.minimum(height, width)
        image = tf.image.crop_to_bounding_box(
            data["image"],
            (height - crop_size) // 2,
            (width - crop_size) // 2,
            crop_size,
            crop_size,
        )

        # resize and clip
        # for image downsampling it is important to turn on antialiasing
        image = tf.image.resize(image, size=[image_size, image_size], antialias=True)
        return ops.clip(image / 255.0, 0.0, 1.0)


    def prepare_dataset(split):
        # the validation dataset is shuffled as well, because data order matters
        # for the KID estimation
        return (
            tfds.load(dataset_name, split=split, shuffle_files=True)
            .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .repeat(dataset_repetitions)
            .shuffle(10 * train_config.batch_size)
            .batch(train_config.batch_size, drop_remainder=True)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
    ####

    # load dataset
    train_dataset = prepare_dataset("train[:80%]+validation[:80%]+test[:80%]")
    val_dataset = prepare_dataset("train[80%:]+validation[80%:]+test[80%:]")




    ###########
    # create and compile the model
    model = DiffusionModel(train_config)
    # below tensorflow 2.9:
    # pip install tensorflow_addons
    # import tensorflow_addons as tfa
    # optimizer=tfa.optimizers.AdamW
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=train_config.learning_rate, weight_decay=train_config.weight_decay
        ),
        loss=keras.losses.mean_absolute_error,
    )
    # pixelwise mean absolute error is used as loss

    # save the best model based on the validation KID metric
    checkpoint_path = f'{train_config.data_name}/Models/lasts/diffusion_model.weights.h5'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="val_kid",
        mode="min",
        save_best_only=True,
    )

    # calculate mean and variance of training dataset for normalization
    model.normalizer.adapt(train_dataset)




    

    # run training and plot generated images periodically
    model.fit(
        train_dataset,
        epochs=train_config.num_epochs,
        validation_data=val_dataset,
        callbacks=[
            checkpoint_callback,
        ],
    )



    # load the best model and generate images
    model.load_weights(checkpoint_path)
    val_iterator = iter(val_dataset)
    model.plot_images(val_iterator.get_next(),plot_diffusion_steps = 50)