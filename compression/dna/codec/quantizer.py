#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 June 03, 16:29:02
@last modified : 2022 June 11, 21:49:28
"""

import numpy as np
import tensorflow as tf
from bisect import bisect
from scipy.stats import norm

def reduce_batch(fn: callable, x: tf.Tensor, *args, **kwargs):
    """
    Decorator to apply a function on the batch axis
    """
    rank = tf.rank(x).numpy()
    assert rank > 1, "The rank of the tensor must be greater than 1"
    axis = range(1, rank)
    return fn(x, axis=axis, *args, **kwargs)

def quantize(x: tf.Tensor, span: float=255):
    """
    Quantize a tensor to a given dtype
    """
    qmin = reduce_batch(tf.reduce_min, x, keepdims=False)
    qmax = reduce_batch(tf.reduce_max, x, keepdims=False)

    return tf.stack([span * (e - mi) / (ma - mi) for e, mi, ma in zip(x, qmin, qmax)], axis=0), tf.stack([qmin, qmax], axis=1)

def dequantize(x: tf.Tensor, ranges: tf.Tensor, span: float=255):
    """
    Dequantize a tensor to a given dtype
    """
    qmin, qmax = ranges[:,0], ranges[:,1]

    return tf.stack([e / span * (ma - mi) + mi for e, mi, ma in zip(x, qmin, qmax)], axis=0)


class NormalQuantizer:
    """
    Quantizer for a normal distribution
    """

    @staticmethod
    def get_ranges(μ, σ, n):
        """
        Get the ranges for the quantization from a normal distribution with μ and σ
        """
        eps = 1/n
        z_ranges = norm.ppf(np.arange(eps, 1+eps, eps))
        return μ + σ * z_ranges

    @classmethod
    def quantize(cls, x):
        """
        Quantize a tensor to a given dtype
        """
        def _quantize_batch(x):
            axes = range(tf.rank(x).numpy())
            μ, σ = tf.nn.moments(x, axes=axes)
            ranges = cls.get_ranges(μ, σ, n=255)
            xf = tf.reshape(x, (-1,))
            yf = tf.map_fn(lambda z: bisect(ranges, z), xf, fn_output_signature=tf.int32)
            return tf.reshape(yf, tf.shape(x)), (μ, σ)

        qx, moments = zip(*[_quantize_batch(e) for e in tf.unstack(x)])
        return tf.convert_to_tensor(qx), tf.convert_to_tensor(moments)
