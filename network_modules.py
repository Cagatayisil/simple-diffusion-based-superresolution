import math
import keras
from keras import layers
from keras import ops

# Global constants
EMBEDDING_DIMS = 32
EMBEDDING_MAX_FREQUENCY = 1000.0
EMBEDDING_MIN_FREQUENCY = 1.0

@keras.saving.register_keras_serializable()
def sinusoidal_embedding(x):
    """
    Create a sinusoidal embedding of the input tensor.

    Args:
        x (tensor): Input tensor to embed.

    Returns:
        tensor: Sinusoidal embedding of the input tensor.
    """
    frequencies = ops.exp(
        ops.linspace(
            ops.log(EMBEDDING_MIN_FREQUENCY),
            ops.log(EMBEDDING_MAX_FREQUENCY),
            EMBEDDING_DIMS // 2,
        )
    )
    angular_speeds = ops.cast(2.0 * math.pi * frequencies, "float32")
    embeddings = ops.concatenate(
        [ops.sin(angular_speeds * x), ops.cos(angular_speeds * x)], axis=3
    )
    return embeddings

def ResidualBlock(width):
    """
    Create a residual block with the specified width.

    Args:
        width (int): Number of filters in the convolutional layers.

    Returns:
        function: A function that applies the residual block to an input tensor.
    """
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x
    return apply

def DownBlock(width, block_depth):
    """
    Create a down-sampling block with the specified width and depth.

    Args:
        width (int): Number of filters in the convolutional layers.
        block_depth (int): Number of residual blocks to apply.

    Returns:
        function: A function that applies the down-sampling block to an input tensor and skip connections.
    """
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x
    return apply

def UpBlock(width, block_depth):
    """
    Create an up-sampling block with the specified width and depth.

    Args:
        width (int): Number of filters in the convolutional layers.
        block_depth (int): Number of residual blocks to apply.

    Returns:
        function: A function that applies the up-sampling block to an input tensor and skip connections.
    """
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x
    return apply

def get_network(opt):
    """
    Create the main network architecture based on the provided options.

    Args:
        opt (object): An object containing network configuration options.

    Returns:
        keras.Model: The constructed network model.
    """
    image_size = opt.image_size
    widths = opt.widths
    block_depth = opt.block_depth

    # Define input layers
    noisy_images = keras.Input(shape=(image_size, image_size, 6))
    noise_variances = keras.Input(shape=(1, 1, 1))

    # Create and upsample embedding
    e = layers.Lambda(sinusoidal_embedding, output_shape=(1, 1, EMBEDDING_DIMS))(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    # Initial convolution and concatenation
    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    # Down-sampling path
    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    # Middle blocks
    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    # Up-sampling path
    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    # Final convolution
    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")