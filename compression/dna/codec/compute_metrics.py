#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 May 20, 10:55:02
@last modified : 2022 May 20, 15:24:57
"""

import hydra
import pickle
import numpy as np
from glob import glob
from functools import partial
from multiprocessing import Pool
from utils import number_of_nucleotides


def nucleotide_rate(x, z, x_hat):
    """
    Compute the nucleotide rate.
    """
    return number_of_nucleotides(z) / num_voxels(x, z, x_hat)


def num_voxels(x, z, x_hat):
    """
    Compute the number of voxels.
    """
    return np.prod(x.shape).astype(np.int32)


def num_occupied_voxels(x, z, x_hat):
    """
    Compute the number of occupied voxels.
    """
    return np.sum(x).astype(np.int32)


@hydra.main(config_path="config/compute_metrics", config_name="default.yaml")
def compute_metrics(cfg):
    def assert_same_files(x, z, x_hat):
        x_name = x.split("/")[-1].split(".")[0]
        z_name = z.split("/")[-1].split(".")[0]
        x_hat_name = x_hat.split("/")[-1].split(".")[0]
        assert (
            x_name == z_name == x_hat_name
        ), "The files are not the same : {} {} {}".format(x, z, x_hat)
        return x_name

    global compute_metrics_worker
    def compute_metrics_worker(x_file, z_file, x_hat_file, cfg):
        x = np.load(x_file)
        with open(z_file, "rb") as f:
            z = pickle.load(f)
        x_hat = np.load(x_hat_file)
        name = assert_same_files(x_file, z_file, x_hat_file)
        return f"{name}," + ",".join(
            [str(globals()[metric_name](x, z, x_hat)) for metric_name in cfg.metrics]
        )

    # Load the data
    x_files = sorted(glob(cfg.io.x.rstrip("/*") + "/*.npy"))
    z_files = sorted(glob(cfg.io.z.rstrip("/*") + "/*.pkl"))
    x_hat_files = sorted(glob(cfg.io.x_hat.rstrip("/*") + "/*.npy"))

    # Compute all metrics for each data point
    with Pool() as p:
        f = partial(compute_metrics_worker, cfg=cfg)
        lines = p.starmap(f, list(zip(x_files, z_files, x_hat_files)))

    header = "name," + ",".join(cfg.metrics)
    with open(cfg.io.csv, "w+") as f:
        f.write(header + "\n")
        f.write("\n".join(lines))


if __name__ == "__main__":
    compute_metrics()
