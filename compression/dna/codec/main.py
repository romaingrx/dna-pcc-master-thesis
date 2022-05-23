#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 May 10, 11:15:58
@last modified : 2022 May 23, 18:35:15
"""

from functools import partial
from itertools import product
from tqdm import tqdm
import hydra
import logging
import numpy as np
import os
import tensorflow as tf
import tensorflow_compression as tfc
from jpegdna.codecs import JpegDNA
from ray.util.multiprocessing import Pool


from helpers import omegaconf2namespace
from layers import AnalysisTransform, SynthesisTransform
from src import pc_io
from src.compression_utilities import (
    pack_tensor,
    unpack_tensor,
    compute_optimal_threshold,
)
from src.focal_loss import focal_loss
from utils import (
    pc_dir_to_ds,
    number_of_nucleotides,
)

# from codec import BatchMultiChannelsJpegDNA

logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class BatchMultiChannelsJpegDNA:
    oligo_length = 200  # Weird but impossible to adapt in Xavier's code

    def __init__(self, alpha):
        self._alpha = alpha

    def encode_batch(self, x):
        """Encode a batch of images with several channels into a list of oligos."""
        # Input shape: (batch, height, width, channels)
        # Output shape: (batch, channels, nb_oligos)
        assert (
            len(x.shape) == 4
        ), "x must be a 4D tensor (batch, height, width, channels)"

        global _encode_worker

        def _encode_worker(x, alpha):
            """Encode a 2 dimensional array into several oligos."""
            return JpegDNA(alpha).encode(x.numpy(), "from_img")

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

        global _decode_worker

        def _decode_worker(oligos, alpha):
            """Decode a list of oligos into a 2 dimensional array."""
            return JpegDNA(alpha).decode(oligos.numpy().astype(str))

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

    def __init__(self, args, name="CompressionModel"):
        super().__init__(name=name)
        self._args = args

        self.prior = tfc.NoisyDeepFactorized(batch_shape=[args.latent_depth])

        self.analysis_transform = AnalysisTransform(args.num_filters, args.latent_depth)
        self.synthesis_transform = SynthesisTransform(args.num_filters)

        if args.transfer_learning_model is not None:

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

                # Have to set manually the weights of the synthesis transform
                # because the weights are not stored in the same order
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
        shape = batch_size, b1, b2, b3, latent_depth = tf.shape(
            x
        )  # TODO change: the way to store the mid shape

        # Quantize the blocks
        quantize_range = (
            self._args.quantize.min,
            self._args.quantize.max,
        )  # TODO change: the way to compute the quantization range
        quantized_x, *_ = tf.quantization.quantize(x, *quantize_range, tf.quint8)

        # Turn the blocks into several images to use the Jpeg DNA codec
        quantized_x = tf.reshape(quantized_x, [batch_size, b1 * b2, b3, latent_depth])

        # Encode the images to DNA oligos
        # logger.info("Encoding the latent blocks to DNA oligos...")
        with tf.device("/cpu:0"):
            codec = BatchMultiChannelsJpegDNA(self._args.alpha)
            oligos = codec.encode_batch(tf.cast(quantized_x, tf.int32))

        return oligos, shape[1:]

    def dna_decoding(self, oligos, shape):
        """Decodes the DNA oligos to latent blocks."""
        assert len(oligos.shape) == 3, (
            f"The input must be of shape [batch_size, latent_depth, nb_oligos] "
            "but received {oligos.shape}."
        )
        batch_size, latent_depth, nb_oligos = oligos.shape
        # Decode the DNA oligos to images
        # logger.info("Decoding the DNA oligos to latent blocks...")
        with tf.device("/cpu:0"):
            codec = BatchMultiChannelsJpegDNA(self._args.alpha)
            quantized_x = codec.decode_batch(oligos)

        quantize_range = (self._args.quantize.min, self._args.quantize.max)

        # Dequantize the blocks
        x, *_ = tf.quantization.dequantize(
            tf.cast(quantized_x, tf.quint8), *quantize_range
        )

        # Turn the images into blocks
        x = tf.reshape(x, [batch_size, *shape])

        return x

    def compress(self, x):
        geo_x = x[:, :, :, :, 0]

        num_voxels = tf.cast(tf.size(geo_x), tf.float32)
        num_occupied_voxels = tf.reduce_sum(geo_x)

        # Build the encoder (analysis) half of the hierarchical autoencoder.
        y = self.analysis_transform(x)

        # Build the bottleneck
        z, shape = self.dna_encoding(y)

        info = {
            "num_voxels": num_voxels,
            "num_occupied_voxels": num_occupied_voxels,
        }

        return z, shape, info

    def decompress(self, z, y_shape):
        # Build the decoder (synthesis) half of the hierarchical autoencoder.
        y_hat = self.dna_decoding(z, y_shape)

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


def load_model(args):
    """Loads the model."""
    if args.model_checkpoint != "":
        return tf.keras.models.load_model(args.model_checkpoint)
    return CompressionModel(args)


def compress(model, args):
    """Compress the dataset"""

    # Create a tensorflow dataset from the point clouds.
    ds = pc_dir_to_ds(
        args.io.input,
        args.blocks.resolution,
        args.blocks.channels_last,
    )

    # Then, compress/decompress the dataset and save the results
    for data in tqdm(ds, total=ds.cardinality().numpy()):
        x = data["input"]
        name = data["fname"].numpy().decode("UTF-8").split("/")[-1].split(".")[0]

        z, y_shape, _ = model.compress(tf.expand_dims(x, 0))
        z_strings = z[0]

        # Calculate the adaptive threshold.
        if args.threshold == "adaptive":
            logger.info("Calculating the adaptive threshold...")
            threshold = compute_optimal_threshold(
                model,
                z_strings,
                y_shape,
                data["pc"].numpy(),
                delta_t=0.01,
                breakpt=150,
                verbose=1,
            )
        else:
            assert 0.0 < args.threshold < 1.0, "Threshold must be between 0 and 1."
            threshold = tf.constant(args.threshold)

        logger.info("Threshold: %f", threshold)
        logger.info("Pack the representations...")
        nucleotide_stream = pack_tensor(
            threshold, BatchMultiChannelsJpegDNA.oligo_length, y_shape, z_strings
        )

        logger.info("Saving the compressed data to %s", args.io.output)

        # Create recursively the output directory if it does not exist.
        os.makedirs(args.io.output, exist_ok=True)
        with open(os.path.join(args.io.output, name + ".dna"), "w+") as f:
            f.write(nucleotide_stream)

        return


def decompress(model, args):
    """Decompress the dataset"""
    global ds

    files = pc_io.get_files(args.io.input)

    for fname in tqdm(files):
        name = fname.split("/")[-1].split(".")[0]
        with open(fname, "r") as fd:
            nucleotide_stream = fd.read()

        global threshold, oligo_length, y_shape, z_strings
        threshold, _, y_shape, z_strings = unpack_tensor(
            nucleotide_stream,
        )

        # Reconstruct the point clouds.
        logger.info("Reconstructing the point clouds...")
        global x_hat
        x_hat = model.decompress(tf.expand_dims(z_strings, 0), y_shape).numpy()[0]

        pa = np.argwhere(x_hat[..., 0] > threshold.numpy()).astype(np.float32)

        # Save the reconstructed point clouds.
        os.makedirs(args.io.output, exist_ok=True)

        pc_io.write_df(args.io.output.rstrip("/*") + f"/{name}.ply", pc_io.pa_to_df(pa))

        return


@hydra.main(config_path="config/main", config_name="default.yaml", version_base="1.2")
def main(cfg):
    global points, ds, model, hist, x, z, x_hat, info, simulator, argss
    args = omegaconf2namespace(cfg)

    if args.task not in ["compress", "decompress"]:
        raise ValueError(f"Unknown task: {args.task}, choose between {tasks}")

    # Load the model
    model = load_model(args.architecture)

    if args.task == "compress":
        compress(model, args.compress)
    elif args.task == "decompress":
        decompress(model, args.decompress)


if __name__ == "__main__":
    # Set the memory growth option so it doesn't allocate all GPUs memory.
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    main()
