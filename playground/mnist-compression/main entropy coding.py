#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 May 06, 14:38:04
@last modified : 2022 May 09, 17:28:51
"""

import tensorflow as tf
import tensorflow_compression as tfc
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


class AutoEncoder(tf.keras.Model):
    def __init__(self, latent_dim, alpha=0.2):
        super(AutoEncoder, self).__init__()
        self._alpha = alpha
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(latent_dim),
            ],
            name="encoder",
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(784, activation="sigmoid"),
                tf.keras.layers.Reshape((28, 28)),
            ],
            name="decoder",
        )

        self._em = tfc.ContinuousBatchedEntropyModel(
            tfc.NoisyDeepFactorized(batch_shape=[latent_dim]),
            coding_rank=2,
            compression=False,
        )

    def compile(self, *args, **kwargs):
        super(AutoEncoder, self).compile(*args, **kwargs)
        self.bitrate = tf.keras.metrics.Mean(name="bitrate")
        self.distortion = tf.keras.metrics.Mean(name="distortion")
        self.loss = tf.keras.metrics.Mean(name="loss")

    def call(self, x, training=False):
        # Compress the images
        latent_representation = self.encoder(x)

        # Use the entropy model
        latent_representation, bits = self._em(latent_representation, training=training)

        # Decompress the images
        reconstructed_images = self.decoder(latent_representation)
        return reconstructed_images, bits / (x.shape[1] * x.shape[2])

    def fit(self, *args, **kwargs):
        retval = super(AutoEncoder, self).fit(*args, **kwargs)
        self._em = tfc.ContinuousBatchedEntropyModel(
            tfc.NoisyDeepFactorized(batch_shape=[self.latent_dim]),
            coding_rank=2,
            compression=True,
        )
        return retval

    def train_step(self, x):
        with tf.GradientTape() as tape:
            reconstructed_images, bitrate = self.call(x, training=True)
            distortion = self.compiled_loss(x, reconstructed_images)
            loss = bitrate + self._alpha * distortion
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.bitrate.update_state(bitrate)
        self.distortion.update_state(distortion)
        self.loss.update_state(loss)
        return {m.name: m.result() for m in [self.bitrate, self.distortion, self.loss]}

    def compress(self, x):
        latent_representation = self.encoder(x)
        return self._em.compress(latent_representation)


model = AutoEncoder(latent_dim=64, alpha=100)
model.compile(optimizer="adam", loss="mse")
model.fit(x_train, epochs=1)
x_test_encoded = model.encoder(x_test)
x_test_decoded = model.decoder(x_test_encoded).numpy()

# Plot the reconstructed images
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test_decoded[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
plt.show()
