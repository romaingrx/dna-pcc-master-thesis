#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 May 13, 16:36:15
@last modified : 2022 May 16, 12:09:33
"""

import os
import tensorflow as tf
from functools import partial
from itertools import product
from jpegdna.codecs import JpegDNA
from concurrent.futures import ProcessPoolExecutor as Pool


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
        return tf.stack(tf.split(y, n_batches))

    def decode_batch(self, x):
        """Decode a batch of oligos into a batch of images with several channels."""
        # Input shape: (batch, channels, nb_oligos)
        # Output shape: (batch, height, width, channels)
        assert len(x.shape) == 3, "x must be a 3D tensor (batch, channels, nb_oligos)"
        n_batch, n_channels, n_oligos = x.shape
        indexes = product(range(n_batch), range(n_channels))
        f = partial(_decode_worker, alpha=self._alpha)
        with Pool() as p:
            y = list(p.map(f, [x[idx] for idx in indexes]))
        # Reshape the tensor to (batch, channels, height, width)
        y = tf.stack(tf.split(y, n_batch))
        # Swap axes to have (batch, height, width, channels)
        return tf.transpose(y, [0, 2, 3, 1])


if __name__ == "__main__":
    # Create a fake batch of images (batch, height, width, channels) with integer values from 0 to 255
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    with tf.device("/cpu:0"):
        codec = BatchMultiChannelsJpegDNA(alpha=0.5)
        x = tf.Variable(tf.random.uniform([4, 32, 32, 3], 0, 255, dtype=tf.int32))
        # Encode the batch
        y = codec.encode_batch(x)
        # Decode the batch
        x_hat = codec.decode_batch(y)
