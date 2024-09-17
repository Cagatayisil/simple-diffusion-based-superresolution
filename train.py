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
    """
    Initialize and parse command-line arguments for the super-resolution model.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='superresolution')
    parser.add_argument('--gpu_id', type=int, default=0, help='the ID of the visible GPU device (only used when not in parallel mode)')
    parser.add_argument('--data_name', type=str, default='SR4_2', help='Name of the dataset')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--image_size', type=int, default=64, help='Image size for data loading')
    parser.add_argument('--down_image_size', type=int, default=16, help='Low-resolution image size for data loading')
    parser.add_argument('--kid_image_size', type=int, default=75, help='Image size for KID metric')
    parser.add_argument('--kid_diffusion_steps', type=int, default=5, help='Number of diffusion steps for KID metric')
    parser.add_argument('--plot_diffusion_steps', type=int, default=20, help='Number of diffusion steps for plotting')
    parser.add_argument('--min_signal_rate', type=float, default=0.02, help='Minimum signal rate for sampling')
    parser.add_argument('--max_signal_rate', type=float, default=0.95, help='Maximum signal rate for sampling')
    
    # Architecture parameters
    parser.add_argument('--embedding_dims', type=int, default=32, help='Embedding dimensions')
    parser.add_argument('--embedding_max_frequency', type=float, default=1000.0, help='Maximum frequency for embedding')
    parser.add_argument('--block_depth', type=int, default=2, help='Depth of residual blocks')
    
    # Optimization parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loading')
    parser.add_argument('--ema', type=float, default=0.999, help='Exponential moving average decay rate')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')

    vc = parser.parse_args()
    vc.widths = [32, 64, 96, 128]

    # Create directories and save configuration
    if not os.path.exists(f'{vc.data_name}'):
        os.makedirs(f'{vc.data_name}')

    vc.config_file = f'{vc.data_name}/config_file.txt'
    cfg = '----------------- Options ---------------\n'
    for k, v in sorted(vars(vc).items()):
        cfg += f'{str(k):>25}: {str(v):<30}\n'
    cfg += '----------------- End -------------------\n'

    print(cfg)
    with open(vc.config_file, 'w') as f:
        f.writelines(f'{cfg}')

    return vc

def preprocess_image(data):
    """
    Preprocess the input image data.
    
    Args:
        data (dict): Input data containing the image
    
    Returns:
        tuple: Processed high-resolution and low-resolution images
    """
    # Center crop image
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

    # Resize and clip
    # For image downsampling it is important to turn on antialiasing
    image = tf.image.resize(image, size=[image_size, image_size], antialias=True)
    downsampled_image = tf.image.resize(image, size=[down_image_size, down_image_size], antialias=True, method='nearest')

    return ops.clip(image / 255.0, 0.0, 1.0), ops.clip(downsampled_image / 255.0, 0.0, 1.0)

def prepare_dataset(split):
    """
    Prepare the dataset for training or validation.
    
    Args:
        split (str): Dataset split to use
    
    Returns:
        tf.data.Dataset: Prepared dataset
    """
    # The validation dataset is shuffled as well, because data order matters
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

def create_and_compile_model(train_config):
    """
    Create and compile the diffusion model.
    
    Args:
        train_config (argparse.Namespace): Training configuration
    
    Returns:
        keras.Model: Compiled diffusion model
    """
    model = DiffusionModel(train_config)
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=train_config.learning_rate, weight_decay=train_config.weight_decay
        ),
        loss=keras.losses.mean_absolute_error,
    )
    return model

def train_model(model, train_dataset, val_dataset, train_config):
    """
    Train the diffusion model.
    
    Args:
        model (keras.Model): The model to train
        train_dataset (tf.data.Dataset): Training dataset
        val_dataset (tf.data.Dataset): Validation dataset
        train_config (argparse.Namespace): Training configuration
    """
    checkpoint_path = f'{train_config.data_name}/Models/lasts/diffusion_model.weights.h5'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="val_kid",
        mode="min",
        save_best_only=True,
    )

    # Calculate mean and variance of training dataset for normalization
    tra_dataset = train_dataset.map(lambda x, y: (x))
    model.normalizer.adapt(tra_dataset)

    # Run training
    model.fit(
        train_dataset,
        epochs=train_config.num_epochs,
        validation_data=val_dataset,
        callbacks=[checkpoint_callback],
    )

    return checkpoint_path

def main():
    """
    Main function to run the super-resolution training process.
    """
    global train_config
    train_config = init_parameters()

    # Set up GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{train_config.gpu_id}"

    # Create necessary directories
    if not os.path.exists(f'{train_config.data_name}/val_images/'):
        os.makedirs(f'{train_config.data_name}/val_images/')
    if not os.path.exists(f'{train_config.data_name}/Models/last/'):
        os.makedirs(f'{train_config.data_name}/Models/last/')

    global image_size, down_image_size, dataset_name, dataset_repetitions
    image_size = train_config.image_size
    down_image_size = train_config.down_image_size
    dataset_name = "oxford_flowers102"
    dataset_repetitions = 5

    # Prepare datasets
    train_dataset = prepare_dataset("train[:80%]+validation[:80%]+test[:80%]")
    val_dataset = prepare_dataset("train[80%:]+validation[80%:]+test[80%:]")

    # Create and train the model
    model = create_and_compile_model(train_config)
    checkpoint_path = train_model(model, train_dataset, val_dataset, train_config)

    # Load the best model and generate images
    model.load_weights(checkpoint_path)
    val_iterator = iter(val_dataset)
    model.plot_images(val_iterator.get_next(), plot_diffusion_steps=50)

if __name__ == '__main__':
    main()