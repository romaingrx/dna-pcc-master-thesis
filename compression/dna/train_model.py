#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 May 10, 11:15:58
@last modified : 2022 May 11, 18:43:07
"""

import pdb

import os
import hydra
import numpy as np
import multiprocessing
import tensorflow as tf
from functools import partial

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src import pc_io
from src import processing
from src.focal_loss import focal_loss
from utils import train_test_split_ds


class Residual_Block(tf.keras.layers.Layer):
    """Residual transform used in the Analysis and Synthesis transform"""

    def __init__(self, num_filters, name):
        super().__init__(name=name)
        self.block1 = tf.keras.layers.Conv3D(
            num_filters / 4, (3, 3, 3), padding="same", activation="relu"
        )

        self.block2 = tf.keras.layers.Conv3D(
            num_filters / 2, (3, 3, 3), padding="same", activation="relu"
        )

        self.block3 = tf.keras.layers.Conv3D(
            num_filters / 4, (1, 1, 1), padding="same", activation="relu"
        )

        self.block4 = tf.keras.layers.Conv3D(
            num_filters / 4, (3, 3, 3), padding="same", activation="relu"
        )

        self.block5 = tf.keras.layers.Conv3D(
            num_filters / 2, (1, 1, 1), padding="same", activation="relu"
        )

        self.concat = tf.keras.layers.Concatenate()
        self.add = tf.keras.layers.Add()

    def call(self, x):
        y1 = self.block1(x)
        y1 = self.block2(y1)

        y2 = self.block3(x)
        y2 = self.block4(y2)
        y2 = self.block5(y2)

        concat = self.concat([y1, y2])
        output = self.add([x, concat])
        return output


class AnalysisTransform(tf.keras.layers.Layer):
    """Analysis transform used to turn the input into its latent representation"""

    def __init__(self, num_filters, latent_depth):
        super().__init__(name="analysis")

        self.conv = tf.keras.layers.Conv3D(
            num_filters, (9, 9, 9), strides=(2, 2, 2), padding="same", activation="relu"
        )

        self.conv_int = tf.keras.layers.Conv3D(
            num_filters, (5, 5, 5), strides=(2, 2, 2), padding="same", activation="relu"
        )

        self.convout = tf.keras.layers.Conv3D(
            latent_depth,
            (5, 5, 5),
            strides=(2, 2, 2),
            padding="same",
            activation="linear",
        )

        self.res_block1 = Residual_Block(num_filters, name="block_1")
        self.res_block2 = Residual_Block(num_filters, name="block_2")

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.res_block1(x)
        x = self.conv_int(x)
        x = self.res_block2(x)
        x = self.convout(x)
        return x


class SynthesisTransform(tf.keras.layers.Layer):
    """Analysis transform used to turn the input into its latent representation"""

    def __init__(self, num_filters):
        super().__init__(name="synthesis")

        self.conv1 = tf.keras.layers.Conv3DTranspose(
            num_filters, (5, 5, 5), strides=(2, 2, 2), padding="same", activation="relu"
        )

        self.conv2 = tf.keras.layers.Conv3DTranspose(
            num_filters, (5, 5, 5), strides=(2, 2, 2), padding="same", activation="relu"
        )

        self.conv3 = tf.keras.layers.Conv3DTranspose(
            1, (9, 9, 9), strides=(2, 2, 2), padding="same", activation="sigmoid"
        )

        self.block1 = Residual_Block(num_filters, name="block_3")
        self.block2 = Residual_Block(num_filters, name="block_4")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.block1(x)
        x = self.conv2(x)
        x = self.block2(x)
        x = self.conv3(x)
        return x


class CompressionModel(tf.keras.Model):
    """Main model class."""

    # def __init__(self, lmbda, alpha, num_filters,
    #              latent_depth, hyperprior_depth,
    #              num_slices, max_support_slices,
    #              num_scales, scale_min, scale_max, transfer_model_path=None):
    # self.lmbda = lmbda
    # self.alpha = alpha
    # self.latent_depth = latent_depth
    # self.num_filters = num_filters
    # self.num_scales = num_scales
    # self.num_slices = num_slices
    # self.max_support_slices = max_support_slices

    # offset = tf.math.log(scale_min)
    # factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
    #     num_scales - 1.)
    # self.scale_fn = lambda i: tf.math.exp(offset + factor * i)

    def __init__(self, args):
        super().__init__()
        self._args = args

        self.analysis_transform = AnalysisTransform(args.num_filters, args.latent_depth)
        self.synthesis_transform = SynthesisTransform(args.num_filters)
        if args.transfer_learning_model is not None:
            import tensorflow_compression as tfc

            # Initialize the weights of the analysis transform
            a = self.analysis_transform.call(tf.random.normal([1, 64, 64, 64, 1]))
            s = self.synthesis_transform.call(a)

            print("Loading transfer learning model...")
            with tf.device("/cpu:0"):
                transfer_model = tf.keras.models.load_model(
                    args.transfer_learning_model
                )

                self.analysis_transform.set_weights(
                    transfer_model.analysis_transform.weights
                )

                # Have to set manually the weights of the synthesis transform because the weights are not stored in the same order
                self.synthesis_transform.block1.set_weights(
                    transfer_model.synthesis_transform.block1.weights
                )
                self.synthesis_transform.block2.set_weights(
                    transfer_model.synthesis_transform.block2.weights
                )
                self.synthesis_transform.conv1.set_weights(
                    transfer_model.synthesis_transform.conv1.weights
                )
                self.synthesis_transform.conv2.set_weights(
                    transfer_model.synthesis_transform.conv2.weights
                )
                self.synthesis_transform.conv3.set_weights(
                    transfer_model.synthesis_transform.conv3.weights
                )
                print("Transfer learning model loaded.")

        self.dna_encoding = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4 * args.latent_dim, activation="relu"),
                tf.keras.layers.Reshape((args.latent_dim, 4)),
                tf.keras.layers.Softmax(axis=-1),
            ]
        )

        self.dna_decoding = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(8 * 8 * 8 * args.latent_depth, activation="relu"),
                tf.keras.layers.Reshape((8, 8, 8, args.latent_depth)),
            ]
        )

        # self.build((None, None, None, None, 1))

        # The call signature of decompress() depends on the number of slices, so we
        # need to compile the function dynamically.
        # self.decompress = tf.function(
        #         input_signature=3 * [tf.TensorSpec(shape=(3,), dtype=tf.int32)] +
        #         (num_slices + 1) * [tf.TensorSpec(shape=(1,), dtype=tf.string)]
        #         )(self.decompress)

    def compile(self, optimizer, loss):
        super().compile(optimizer=optimizer, loss=loss)
        self.focal_loss = tf.metrics.Mean(name="focal_loss")

    def call(self, x):
        """Computes distortion loss."""

        geo_x = x[:, :, :, :, 0]

        num_voxels = tf.cast(tf.size(geo_x), tf.float32)
        num_occupied_voxels = tf.reduce_sum(geo_x)

        # Build the encoder (analysis) half of the hierarchical autoencoder.
        y = self.analysis_transform(x)

        # Build the bottleneck
        z = self.dna_encoding(y)

        # Reconstruct the image from the bottleneck
        y_hat = self.dna_decoding(z)

        # Build the decoder (synthesis) half of the hierarchical autoencoder.
        x_hat = self.synthesis_transform(y_hat)

        # Compute the focal loss and/or color loss across pixels.
        # Don't clip or round pixel values while training.
        fcl = focal_loss(x, x_hat, gamma=2, alpha=self._args.alpha) / num_voxels
        return fcl, num_occupied_voxels

    def train_step(self, x):
        """Performs a training step."""
        with tf.GradientTape() as tape:
            loss, num_occupied_voxels = self(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.focal_loss.update_state(loss)
        return {m.name: m.result() for m in [self.focal_loss]}

    def test_step(self, x):
        """Performs a test step."""
        loss, num_occupied_voxels = self(x)
        self.focal_loss.update_state(loss)
        return {m.name: m.result() for m in [self.focal_loss]}


@hydra.main(config_name="config.yaml", config_path=".")
def main(args):
    global points, ds, model, hist

    # Load the point clouds
    files = pc_io.get_files(args.io.input)
    # Load the blocks from the files.
    p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(
        args.blocks.resolution, args.blocks.channels_last
    )
    points = pc_io.load_points(files, p_min, p_max)

    with tf.device("CPU"):
        with multiprocessing.Pool() as pool:
            # Transform the point clouds to tensors.
            points = pool.map(
                partial(
                    processing.pc_to_tf,
                    dense_tensor_shape=dense_tensor_shape,
                    channels_last=args.blocks.channels_last,
                ),
                points,
            )
            # Convert the sparse tensors to dense tensors.
            points = pool.map(
                partial(processing.process_x, dense_tensor_shape=dense_tensor_shape),
                points,
            )

    # Create a tensorflow dataset from the point clouds.
    ds = (
        tf.data.Dataset.from_tensor_slices(points)
        .shuffle(len(points))
        .batch(args.train.batch_size)
    )

    # Train, test split the dataset.
    train_ds, validation_ds = train_test_split_ds(
        ds, validation_split=args.train.validation_split
    )

    print(f"Training on {train_ds.cardinality().numpy()} batches with a {args.train.batch_size} batch size.")
    print(f"Validating on {validation_ds.cardinality().numpy()} batches with a {args.train.batch_size} batch size.")

    # Load the model on multi worker GPUs.
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # Create the model.
        model = CompressionModel(args.model)

        # Compile the model.
        model.compile(
            optimizer="adam",
            loss=None,
        )

    # Train the model.
    hist = model.fit(
        train_ds.prefetch(-1),
        validation_data=validation_ds.prefetch(-1),
        epochs=args.train.epochs,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=args.train.log_dir,
                update_freq="epoch",
                histogram_freq=1,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ],
    )


if __name__ == "__main__":
    # Set the memory growth option so it doesn't allocate all GPUs memory.
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    main()
