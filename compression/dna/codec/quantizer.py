#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 June 03, 16:29:02
@last modified : 2022 June 03, 17:01:19
"""

import tensorflow as tf

def reduce_batch(fn: callable, x: tf.Tensor, *args, **kwargs):
    """
    Decorator to apply a function on the batch axis
    """
    rank = tf.rank(x).numpy()
    assert rank > 1, "The rank of the tensor must be greater than 1"
    axis = range(1, rank)
    return fn(x, axis=axis, *args, **kwargs)

def quantize(x: tf.Tensor):
    """
    Quantize a tensor to a given dtype
    """
    qmin = reduce_batch(tf.reduce_min, x, keepdims=False)
    qmax = reduce_batch(tf.reduce_max, x, keepdims=False)

    return tf.stack([255 * (e - mi) / (ma - mi) for e, mi, ma in zip(x, qmin, qmax)], axis=0), tf.stack([qmin, qmax], axis=1)

def dequantize(x: tf.Tensor, ranges: tf.Tensor):
    """
    Dequantize a tensor to a given dtype
    """
    qmin, qmax = ranges[:,0], ranges[:,1]

    return tf.stack([e / 255 * (ma - mi) + mi for e, mi, ma in zip(x, qmin, qmax)], axis=0)
