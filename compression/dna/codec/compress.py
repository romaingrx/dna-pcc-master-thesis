#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 May 10, 11:15:58
@last modified : 2022 May 17, 15:03:34
"""

from sys import settrace

import os
import hydra
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
import tensorflow as tf
from functools import partial
from jpegdna.codecs import JpegDNA

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src import pc_io
from src import processing
from utils import dir_to_ds, number_of_nucleotides, train_test_split_ds
# from codec import BatchMultiChannelsJpegDNA
from src.focal_loss import focal_loss
from layers import AnalysisTransform, SynthesisTransform

logger = logging.getLogger(__name__)

from itertools import product
from ray.util.multiprocessing import Pool

def _encode_worker(x, alpha):
    """Encode a 2 dimensional array into several oligos."""
    return JpegDNA(alpha).encode(x.numpy(), "from_img")


def _decode_worker(oligos, alpha):
    """Decode a list of oligos into a 2 dimensional array."""
    return JpegDNA(alpha).decode(oligos.numpy().astype(str))


class BatchMultiChannelsJpegDNA:
    def __init__(self, alpha):
        self._alpha = alpha

    def encode_batch(self, x):
        """Encode a batch of images with several channels into a list of oligos."""
        # Input shape: (batch, height, width, channels)
        # Output shape: (batch, channels, nb_oligos)
        assert (
            len(x.shape) == 4
        ), "x must be a 4D tensor (batch, height, width, channels)"
        # Input shape: (batch, height, width, channels)
        # Output shape: (batch, channels, nb_oligos)
        n_batches, n_channels = x.shape[0], x.shape[3]
        indexes = list(product(range(n_batches), range(n_channels)))
        f = partial(_encode_worker, alpha=self._alpha)
        with Pool() as p:
            y = list(p.map(f, [x[i, :, :, j] for i, j in indexes]))

        # Reshape the tensor to (batch, channels, nb_oligos)
        y = [
            [y[i * n_channels + j] for j in range(n_channels)] for i in range(n_batches)
        ]
        return tf.ragged.constant(y, dtype=tf.string)

    def decode_batch(self, x):
        """Decode a batch of oligos into a batch of images with several channels."""
        # Input shape: (batch, channels, nb_oligos)
        # Output shape: (batch, height, width, channels)
        assert len(x.shape) == 3, "x must be a 3D tensor (batch, channels, nb_oligos)"
        n_batch, n_channels, _ = (
            x.bounding_shape() if type(x) == tf.RaggedTensor else x.shape
        )
        indexes = product(range(n_batch), range(n_channels))
        f = partial(_decode_worker, alpha=self._alpha)
        with Pool() as p:
            y = list(p.map(f, [x[i, j] for i, j in indexes]))
        y = tf.convert_to_tensor(y)
        # Reshape the tensor to (batch, channels, height, width)
        y = tf.stack(tf.split(y, n_batch.numpy()))
        # Swap axes to have (batch, height, width, channels)
        return tf.transpose(y, [0, 2, 3, 1])


class CompressionModel(tf.keras.Model):
    """Main model class."""

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

            logger.info("Loading transfer learning model...")
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
                logger.info("Transfer learning model loaded.")

    def compile(self, optimizer, loss):
        super().compile(optimizer=optimizer, loss=loss)
        self.focal_loss = tf.metrics.Mean(name="focal_loss")
        self.nucleotides_rate = tf.metrics.Mean(name="nucleotides_rate")

    def dna_encoding(self, x):
        """Encodes the latent blocks to DNA oligos."""
        assert (
            len(x.shape) == 5
        ), "The input must be of shape [batch_size, b1, b2, b3, latent_depth]."
        self._shape = batch_size, b1, b2, b3, latent_depth = tf.shape(
            x
        )  # TODO change: the way to store the mid shape

        # Quantize the blocks
        self._quantize_range = (
            tf.reduce_min(x),
            tf.reduce_max(x),
        )  # TODO change: the way to store the range
        quantized_x, *_ = tf.quantization.quantize(x, *self._quantize_range, tf.quint8)

        # Turn the blocks into several images to use the Jpeg DNA codec
        quantized_x = tf.reshape(quantized_x, [batch_size, b1 * b2, b3, latent_depth])

        # Encode the images to DNA oligos
        # logger.info("Encoding the latent blocks to DNA oligos...")
        with tf.device("/cpu:0"):
            codec = BatchMultiChannelsJpegDNA(self._args.alpha)
            oligos = codec.encode_batch(tf.cast(quantized_x, tf.int32))

        return oligos

    def dna_decoding(self, oligos):
        """Decodes the DNA oligos to latent blocks."""
        assert (
            len(oligos.shape) == 3
        ), "The input must be of shape [batch_size, latent_depth, nb_oligos]."
        batch_size, latent_depth, nb_oligos = oligos.shape
        # Decode the DNA oligos to images
        # logger.info("Decoding the DNA oligos to latent blocks...")
        with tf.device("/cpu:0"):
            codec = BatchMultiChannelsJpegDNA(self._args.alpha)
            quantized_x = codec.decode_batch(oligos)


        # Dequantize the blocks
        x, *_ = tf.quantization.dequantize(tf.cast(quantized_x, tf.quint8), *self._quantize_range)

        # Turn the images into blocks
        x = tf.reshape(x, [batch_size, *self._shape[1:]])
        
        return x

    def compress(self, x):
        geo_x = x[:, :, :, :, 0]

        num_voxels = tf.cast(tf.size(geo_x), tf.float32)
        num_occupied_voxels = tf.reduce_sum(geo_x)

        # Build the encoder (analysis) half of the hierarchical autoencoder.
        y = self.analysis_transform(x)

        # Build the bottleneck
        z = self.dna_encoding(y)

        info = {
            "num_voxels": num_voxels,
            "num_occupied_voxels": num_occupied_voxels,
        }

        return z, info

    def decompress(self, z):
        # Build the decoder (synthesis) half of the hierarchical autoencoder.
        y_hat = self.dna_decoding(z)

        # Build the bottleneck
        x_hat = self.synthesis_transform(y_hat)

        return x_hat

    def call(self, x):
        """Computes distortion loss."""

        # Compute the bottleneck
        z, info = self.compress(x)

        x_hat = self.decompress(z)

        # Compute the focal loss and/or color loss across pixels.
        # Don't clip or round pixel values while training.
        fcl = focal_loss(x, x_hat, gamma=2, alpha=self._args.alpha) / num_voxels
        nucleotide_rate = number_of_nucleotides(z) / num_voxels
        loss = nucleotide_rate + self._args.lmbda * fcl
        info = {
            **info,
            "focal_loss": fcl,
            "nucleotide_rate": nucleotide_rate,
        }
        return loss, info

    def train_step(self, x):
        raise NotImplementedError

    def test_step(self, x):
        """Performs a test step."""
        loss, info = self.call(x)
        self.focal_loss.update_state(loss)
        return {m.name: m.result() for m in [self.focal_loss]}


@hydra.main(config_name="config.yaml", config_path=".")
def main(args):
    global points, ds, model, hist, x, z, x_hat, info

    # Create a tensorflow dataset from the point clouds.
    ds = dir_to_ds(args.io.input, args.blocks.resolution, args.blocks.channels_last)
    # ds = ds.shuffle(ds.cardinality()).batch(args.train.batch_size)

    # Train, test split the dataset.
    # train_ds, validation_ds = train_test_split_ds(
    #     ds, validation_split=args.train.validation_split
    # )
    # logger.info(
    #     f"Training on {train_ds.cardinality().numpy()} batches with a {args.train.batch_size} batch size."
    # )
    # logger.info(
    #     f"Validating on {validation_ds.cardinality().numpy()} batches with a {args.train.batch_size} batch size."
    # )

    # Create the model.
    model = CompressionModel(args.model)

    files = os.listdir(args.io.input[:-1])
    for x, fname in tqdm(zip(ds, files), total=len(files)):
        name = fname.split(".")[0]
        np.save(f"{args.io.output}/x/{name}.npy", x)
        z, info = model.compress(tf.expand_dims(x, 0))
        np.save(f"{args.io.output}/z/{name}.npy", z[0])
        x_hat = model.decompress(z)[0]
        np.save(f"{args.io.output}/x_hat/{name}.npy", x_hat)



if __name__ == "__main__":
    # Set the memory growth option so it doesn't allocate all GPUs memory.
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    main()
