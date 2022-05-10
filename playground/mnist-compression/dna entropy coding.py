#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 May 06, 14:38:04
@last modified : 2022 May 09, 18:51:44
"""

import multiprocessing
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


@tf.autograph.experimental.do_not_convert
def classes_to_dna(classes):
    fn = lambda x: ["A", "C", "G", "T"][x]
    return tf.map_fn(fn, classes, dtype=tf.string)


@tf.autograph.experimental.do_not_convert
def dna_to_classes(dna):
    fn = lambda x: ["A", "C", "G", "T"].index(x)
    return tf.map_fn(fn, dna, dtype=tf.int32)


class AutoEncoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(latent_dim, activation="relu"),
                tf.keras.layers.Dense(4 * latent_dim),
                tf.keras.layers.Reshape((latent_dim, 4)),
                tf.keras.layers.Softmax(axis=-1),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                # Flat the latent representation
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(784, activation="sigmoid"),
                tf.keras.layers.Reshape((28, 28)),
            ]
        )

    def compile(self, *args, **kwargs):
        super(AutoEncoder, self).compile(*args, **kwargs)
        self.loss = tf.keras.metrics.Mean(name="loss")

    def call(self, x):
        # Compress the images
        latent_representation = self.encoder(x)
        # Decompress the images
        reconstructed_images = self.decoder(latent_representation)

        return reconstructed_images

    def fit(self, *args, **kwargs):
        retval = super(AutoEncoder, self).fit(*args, **kwargs)
        return retval

    def train_step(self, x):
        with tf.GradientTape() as tape:
            latent_representation = self.encoder(x)
            reconstructed_images = self.decoder(latent_representation)
            loss = self.compiled_loss(x, reconstructed_images)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        with tf.GradientTape() as tapeDecoder:
            reconstructed_images = self.decoder(
                K.one_hot(K.argmax(latent_representation, axis=-1), 4)
            )
            loss = self.compiled_loss(x, reconstructed_images)

        gradients = tapeDecoder.gradient(loss, self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.decoder.trainable_variables))

        self.loss.update_state(loss)
        return {m.name: m.result() for m in [self.loss]}

    def compress(self, x):
        # Compress the images
        latent_representation = self.encoder(x)
        # Get the maximum argument of the softmax activation
        latent_argmax = tf.argmax(latent_representation, axis=-1, output_type=tf.int32)
        # Translate the dna code into a string
        with multiprocessing.Pool() as pool:
            dna_code = pool.map(classes_to_dna, latent_argmax)
        return tf.Variable(dna_code, dtype=tf.string)

    def decompress(self, x):
        # Translate the dna code into the categorical representation
        with multiprocessing.Pool() as pool:
            classes = pool.map(dna_to_classes, x)
        latent_classes = tf.Variable(classes, dtype=tf.int32)
        # Transform the tensor to a one-hot vector
        latent_representation = tf.one_hot(latent_classes, 4)
        # Decompress the images
        reconstructed_images = self.decoder(latent_representation)
        return reconstructed_images


model = AutoEncoder(latent_dim=64)
model.compile(optimizer="adam", loss="mse")
model.fit(x_train, epochs=1)

continuous_reconstructed = model.call(x_test)
with tf.device("cpu"):
    dna_reconstructed = model.decompress(model.compress(x_test))

# Plot the reconstructed images
import matplotlib.pyplot as plt

# Plot a single example from one-hot encoded dna
plt.figure(figsize=(10, 10))
for i in range(1, 6):
    plt.subplot(3, 5, i * 3)
    plt.title("Original")
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.subplot(3, 5, (i * 3) + 1)
    plt.title("Continuous")
    plt.imshow(continuous_reconstructed[i].reshape(28, 28), cmap="gray")
    plt.subplot(3, 5, (i * 3) + 2)
    plt.title("DNA")
    plt.imshow(dna_reconstructed[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
plt.show()
