#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 June 03, 19:23:10
@last modified : 2022 June 03, 20:12:21
"""

import os
import numpy as np
from threading import Thread

from src import pc_io


def save_all_intermediate_results(names, ys, y_hats, pas, directory, join_thread=False):
    """
    Save all results in a directory.
    """
    global _save_all_results_runner
    def _save_all_results_runner(name, ys, y_hats, pas, directory):
        for name, y, y_hat, pa in zip(names, ys, y_hats, pas):
            # y
            output_dir = os.path.join(directory, "y")
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, name + ".npy"), y.numpy())
    
            # y_hat
            output_dir = os.path.join(directory, "y_hat")
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, name + ".npy"), y_hat.numpy())
    
            # x_hat
            output_dir = os.path.join(directory, "x_hat")
            os.makedirs(output_dir, exist_ok=True)
            pc_io.write_df(os.path.join(output_dir, name + ".ply"), pc_io.pa_to_df(pa))

    t = Thread(target=_save_all_results_runner, args=(names, ys, y_hats, pas, directory), daemon=False)
    t.start()
    if join_thread:
        t.join()

def save_oligos(names, oligos, directory, join_thread=False):
    """
    Save all oligos in a directory.
    """
    global _save_oligos_runner
    def _save_oligos_runner(names, oligos, directory):
        for name, oligo in zip(names, oligos):
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, name), "w+") as f:
                f.write(oligo)
    
    t = Thread(target=_save_oligos_runner, args=(names, oligos, directory), daemon=False)
    t.start()
    if join_thread:
        t.join()
