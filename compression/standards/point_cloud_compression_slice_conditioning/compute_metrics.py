#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 May 20, 10:55:02
@last modified : 2022 May 31, 11:10:12
"""

from functools import partial, wraps
from glob import glob
from helpers import Namespace, omegaconf2namespace
from multiprocessing import Pool
from src import pc_io, processing
from src.compression_utilities import po2po 
from tqdm import tqdm
from utils import number_of_nucleotides as _number_of_nucleotides
import hydra
import logging
import numpy as np
import os
import pickle
import tensorflow as tf

logger = logging.getLogger(__name__)

def load_point_clouds_to_ds(directory, args, extension='ply'):
    """
    Load point clouds to a list of namespaces.
    """
    files = sorted(glob(directory.rstrip('/*') + f'/*.{extension}'))
    fnames = [file.split('/')[-1].split('.')[0] for file in files]

    p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(args.resolution, args.channel_last)
    point_clouds = pc_io.load_points(files, p_min, p_max)

    with tf.device("CPU"):
        logger.info("Transforming the point clouds to tensors")
        points = [
                processing.pc_to_tf(pc, dense_tensor_shape, args.channel_last)
                for pc in tqdm(point_clouds)
                ]

        # Convert the sparse tensors to dense tensors.
        logger.info("Transforming the sparse tensors to dense ones")
        points = [processing.process_x(pc, dense_tensor_shape) for pc in tqdm(points)]

    return [Namespace({'name':name, 'pc':pc, 'voxel_grid':voxel_grid}) for name, pc, voxel_grid in zip(fnames, point_clouds, points)]

def load_binary_compressed(directory, extension='tfci'):
    files = sorted(glob(directory.rstrip('/*') + f'/*.{extension}'))
    fnames = [file.split('/')[-1].split('.')[0] for file in files]
    return [Namespace({'name':name, 'file':file}) for name, file in zip(fnames, files)]

def ensure_scalar(fn):
    """
    Ensure that the function returns a scalar.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        ret = fn(*args, **kwargs)
        if isinstance(ret, tf.Tensor):
            return ret.numpy()
        return ret
    return wrapper


@ensure_scalar
def number_of_bits(x, z, x_hat):
    """
    Compute the number of bits for a given file.
    """
    return os.stat(z.file).st_size * 8


@ensure_scalar
def bitrate(x, z, x_hat):
    """
    Compute the bitrate for a given file.
    """
    return number_of_bits(x, z, x_hat) / num_occupied_voxels(x, z, x_hat)


@ensure_scalar
def num_voxels(x, z, x_hat):
    """
    Compute the number of voxels.
    """
    return np.prod(x.voxel_grid.shape).astype(np.int32)


@ensure_scalar
def num_occupied_voxels(x, z, x_hat):
    """
    Compute the number of occupied voxels.
    """
    return np.sum(x.voxel_grid).astype(np.int32)

@ensure_scalar
def point_to_point_D1(x, z, x_hat):
    """
    Compute the point to point mse.
    """
    return po2po(x.pc, x_hat.pc)



@hydra.main(config_path="config/compute_metrics", config_name="default.yaml")
def compute_metrics(cfg):
    global args
    args = omegaconf2namespace(cfg)
    def assert_same_files(x_name, z_name, x_hat_name):
        assert (
                x_name == z_name == x_hat_name
                ), "The files are not the same : {} {} {}".format(x_name, z_name, x_hat_name)
        return x_name

    global compute_metrics_worker
    def compute_metrics_worker(x, z, x_hat, cfg):
        name = assert_same_files(x.name, z.name, x_hat.name)
        return f"{name}," + ",".join(
                [str(globals()[metric_name](x, z, x_hat)) for metric_name in cfg.metrics]
                )

    global x_ds, z_ds, x_hat_ds
    x_ds = load_point_clouds_to_ds(cfg.io.x.rstrip("/*") + "/*", args.blocks)
    z_ds = load_binary_compressed(cfg.io.z.rstrip("/*") + "/*")
    x_hat_ds = load_point_clouds_to_ds(cfg.io.x_hat.rstrip("/*") + "/*", args.blocks)

    # Compute all metrics for each data point
    f = partial(compute_metrics_worker, cfg=cfg)
    lines = [f(x, z, x_hat) for x, z, x_hat in tqdm(list(zip(x_ds, z_ds, x_hat_ds)), desc="Computing metrics")]

    header = "name," + ",".join(cfg.metrics)
    with open(cfg.io.csv, "w") as f:
        f.write(header + "\n")
        f.write("\n".join(sorted(lines)))



if __name__ == "__main__":
    compute_metrics()
