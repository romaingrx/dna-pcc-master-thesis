#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 May 13, 11:24:49
@last modified : 2022 June 05, 23:40:27
"""

import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from functools import wraps
from src import pc_io, processing

logger = logging.getLogger(__name__)

extract_name = lambda fname: fname.split('/')[-1].split('.')[0]
extract_ext = lambda fname: fname.split('/')[-1].split('.')[-1]
extract_path = lambda fname: '/'.join(fname.split('/')[:-1])


def pc_dir_to_ds(input_dir, resolution, channels_last, n_files=None, except_filenames=None):
    """Load all point clouds from the input_dir and transform them to a tensorflow dataset."""
    # Load the point clouds
    files = pc_io.get_files(input_dir)
    if except_filenames is not None:
        # Exctract names of except_filenames
        except_filenames = list(map(extract_name, except_filenames))
        files = [file for file in files if extract_name(file) not in except_filenames]
    if len(files) == 0:
        raise ValueError(f"No files found in {input_dir} with except_filenames={except_filenames}")
    if n_files is not None:
        files = files[:n_files]
    files = sorted(files)

    # Load the blocks from the files.
    p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(resolution, channels_last)
    raw_points = pc_io.load_points(files, p_min, p_max)

    with tf.device("CPU"):
        logger.info("Transforming the point clouds to tensors")
        points = [
            processing.pc_to_tf(pc, dense_tensor_shape, channels_last)
            for pc in tqdm(raw_points)
        ]

        # Convert the sparse tensors to dense tensors.
        logger.info("Transforming the sparse tensors to dense ones")
        points = [processing.process_x(pc, dense_tensor_shape) for pc in tqdm(points)]

    # Create a tensorflow dataset from the point clouds.
    ds = tf.data.Dataset.from_tensor_slices(
        {"input": points, "fname": files, "pc": tf.ragged.constant(raw_points)}
    )
    return ds


def number_of_nucleotides(x):
    """Return the number of nucleotides in the latent space."""
    # In general all oligos are the same length (200) but it is preferable to compute the number of nucleotides with the length of each one.
    return tf.reduce_sum(
            [len(elem) for elem in np.reshape(x.numpy(), (-1,))]
    )


def train_test_split_ds(ds, validation_split=0.2, test_split=0.0):
    assert 0 <= validation_split <= 1, "validation_split must be between 0 and 1"
    assert 0 <= test_split <= 1, "test_split must be between 0 and 1"
    assert (
        validation_split + test_split <= 1
    ), "validation_split + test_split must be less than 1"

    n = ds.cardinality().numpy()

    validation_size = int(n * validation_split)
    test_size = int(n * test_split)
    train_size = max(n - test_size - validation_size, 0)

    train_ds = ds.take(train_size)
    validation_ds = ds.skip(train_size)

    if test_split > 0:
        test_ds = validation_ds.skip(validation_size)
        validation_ds = validation_ds.take(validation_size)
        return train_ds, validation_ds, test_ds
    return train_ds, validation_ds


def n_dimensional(fun: callable):
    """
    Decorator to apply a function with a flat array from a n-dimensional array.
    """

    @wraps(fun)
    def wrapper(*args, **kwargs):
        # Get the array from the kwargs
        arg = kwargs.pop("from_arg", "array")
        array = kwargs.pop(arg)
        if array is None:
            raise ValueError(
                f"{arg} is not defined, when using n_dimensional you have to define the argument to get the array from."
            )

        # Flat the array
        flat_array = np.reshape(array, (-1,))
        # Apply the function to the flat array
        y = fun(*args, **{**kwargs, arg: flat_array})
        # Reshape the result
        return np.reshape(y, np.shape(array))

    return wrapper

