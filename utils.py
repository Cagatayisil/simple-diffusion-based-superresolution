import matplotlib.pyplot as plt
import math
import keras
from keras import layers
from keras import ops
from network_modules import *

@keras.saving.register_keras_serializable()
class KID(keras.metrics.Metric):
    """
    Kernel Inception Distance (KID) metric implementation.
    
    KID measures the similarity between two distributions of images.
    It is computed using features extracted from the Inception v3 network.
    """

    def __init__(self, opt, name, **kwargs):
        """
        Initialize the KID metric.

        Args:
            opt: An object containing configuration options.
            name: Name of the metric.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.image_size = opt.image_size
        self.kid_image_size = opt.kid_image_size
        
        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")
        
        # Initialize the Inception v3 encoder
        self._init_encoder()

    def _init_encoder(self):
        """
        Initialize the Inception v3 encoder used for feature extraction.
        """
        self.encoder = keras.Sequential([
            keras.Input(shape=(self.image_size, self.image_size, 3)),
            layers.Rescaling(255.0),
            layers.Resizing(height=self.kid_image_size, width=self.kid_image_size),
            layers.Lambda(keras.applications.inception_v3.preprocess_input),
            keras.applications.InceptionV3(
                include_top=False,
                input_shape=(self.kid_image_size, self.kid_image_size, 3),
                weights="imagenet",
            ),
            layers.GlobalAveragePooling2D(),
        ], name="inception_encoder")

    def polynomial_kernel(self, features_1, features_2):
        """
        Compute the polynomial kernel between two sets of features.

        Args:
            features_1: First set of features.
            features_2: Second set of features.

        Returns:
            Polynomial kernel matrix.
        """
        feature_dimensions = ops.cast(ops.shape(features_1)[1], dtype="float32")
        return (features_1 @ ops.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        """
        Update the state of the KID metric.

        Args:
            real_images: Batch of real images.
            generated_images: Batch of generated images.
            sample_weight: Optional sample weights (not used in this implementation).
        """
        # Extract features using the Inception v3 encoder
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # Compute polynomial kernels
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(generated_features, generated_features)
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # Estimate the squared maximum mean discrepancy
        batch_size = real_features.shape[0]
        batch_size_f = ops.cast(batch_size, dtype="float32")
        
        mean_kernel_real = ops.sum(kernel_real * (1.0 - ops.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = ops.sum(
            kernel_generated * (1.0 - ops.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = ops.mean(kernel_cross)

        # Compute KID
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # Update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        """
        Compute the final KID metric result.

        Returns:
            The average KID value.
        """
        return self.kid_tracker.result()

    def reset_state(self):
        """
        Reset the state of the KID metric.
        """
        self.kid_tracker.reset_state()