#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 June 04, 09:08:42
@last modified : 2022 June 04, 10:01:41
"""

import numpy as np
import pandas as pd
from pyntcloud import PyntCloud


def create_from_inequality(width: int, formula: callable) -> PyntCloud:
    ranger = np.arange(-width//2, width//2)
    x, y, z = np.meshgrid(ranger, ranger, ranger)
    occupancy_map = formula(x, y, z)
    points = np.argwhere(occupancy_map)
    df = pd.DataFrame(
            data={
                'x': points[:, 0],
                'y': points[:, 1],
                'z': points[:, 2]}, dtype=np.float32)
    return PyntCloud(df)


def create_sphere(width: int, radius: int):
    formula = lambda x, y, z: x**2 + y**2 + z**2 <= radius**2
    return create_from_inequality(width, formula)


def create_bounded_sphere(width: int, radius_in: int, radius_out: int):
    gt = lambda x, y, z: x**2 + y**2 + z**2 >= radius_in**2
    lt = lambda x, y, z: x**2 + y**2 + z**2 <= radius_out**2
    formula = lambda x, y, z: np.bitwise_and(gt(x, y, z), lt(x, y, z))
    return create_from_inequality(width, formula)

if __name__ == '__main__':
    dataset_dir = "datasets/tiny"
    sphere = create_bounded_sphere(width=128, radius_in=63, radius_out=64)
    sphere.to_file(f'{dataset_dir}/bounded_sphere.ply')
