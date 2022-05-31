#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 May 13, 15:47:09
@last modified : 2022 May 29, 18:35:38
"""

import os
from glob import glob
from tqdm import tqdm
from functools import partial

import hydra
from omegaconf import OmegaConf
from helpers import omegaconf2namespace, Namespace

import numpy as np
from pyntcloud import PyntCloud
from multiprocessing import Pool
from jpegdna.codecs import JpegDNA

from main import CompressionModel
from src.pc_io import load_pc, get_shape_data
from src.processing import pc_to_occupancy_grid


extract_name = lambda fname: fname.split('/')[-1].split('.')[0]
extract_ext = lambda fname: fname.split('/')[-1].split('.')[-1]
extract_path = lambda fname: '/'.join(fname.split('/')[:-1])

def align_files(*directories):
    """
    Allign all files based on the name of each file and zip them.
    """
    assert all(len(directory) > 0 for directory in directories), "Need at least 1 element in each directory"

    paths = [extract_path(directory[0]) for directory in directories]
    extentions = [extract_ext(directory[0]) for directory in directories] # Assume same extension for all files in a directory
    names = [set([extract_name(fname) for fname in directory]) for directory in directories]
    common_names = set.intersection(*names)

    return [[f'{path}/{name}.{ext}' for name in common_names] for path, ext in zip(paths, extentions)]

def load_file(fname):
    """
    Load a file and return the good format.
    """
    ext = extract_ext(fname)
    if ext == 'npy':
        return np.load(fname)
    elif ext == 'pkl':
        return pickle.load(open(fname, 'rb'))
    elif ext == 'ply':
        return PyntCloud.from_file(fname).points
    else:
        raise Exception(f'Unknown extension {ext}')

def lazy_loader(files, args):
    """
    Load files in a lazy fashion.
    """
    for slice_files in files:
        yield np.squeeze([load_file(fname) for fname in slice_files])

def loader(files, args):
    """
    Load files in cache
    """
    return [load_file(fname) for fname in files]

def load_io_files(args, exception=[]):
    """
    Load all files contained in the io subflag.
    """
    raw_files = [glob(f"{directory}/*") for name, directory in args.io.items() if name not in exception]
    files = align_files(*raw_files)
    return lazy_loader(zip(*files) if len(files)>1 else np.reshape(files, (-1, 1)), args)

# All tasks

def play(args):
    """
    Do not do anything, just play.
    """
    pass

def bypass_all(args):
    """
    Bypass the compression into oligos and directly reconstruct the point clouds
    """
    global x, model, occupancy_grids
    exception = list(args.io.keys())
    exception.remove('x')

    with Pool() as pool:
        point_clouds = [pc for pc in load_io_files(args, exception=exception)]
        f = partial(pc_to_occupancy_grid, resolution=args.blocks.resolution, channel_last=args.blocks.channel_last)
        occupancy_grids = np.array(list(tqdm(pool.imap(f, point_clouds), total=len(point_clouds), desc='Occupancy grid')))

    model = CompressionModel(args.architecture)

def quantization_tables(args):
    """
    See the impact of the quantization tables on the compression
    """
    from jpegdna.codecs import JPEGDNAGray

    def encode_decode_mse(x, gammas=None, gammas_chroma=None):
        if gammas is not None:
            JPEGDNAGray.GAMMAS = gammas
        if gammas_chroma is not None:
            JPEGDNAGray.GAMMAS_CHROMA = gammas_chroma
        codec = JpegDNA(1)
        oligos = codec.encode(x, "from_img")
        reconstructed = codec.decode(oligos)
        return np.mean(np.power(x - reconstructed, 2)), len(np.reshape(oligos, (-1,)))

    inp = np.random.randint(0, 255, size=(64, 64))

    print(f'MSE with default parameters: {encode_decode_mse(inp)}')

    print(f'MSE with ones: {encode_decode_mse(inp, gammas=np.ones((8, 8)), gammas_chroma=np.ones((8, 8)))}')



def quantize(args):
    """
    Quantize the float latent representation into quint8
    """
    pass




@hydra.main(config_path="config/benchmarks", config_name='config.yaml', version_base="1.2")
def main(cfg: OmegaConf) -> None:
    global args
    args = omegaconf2namespace(cfg)

    tasks = ["bypass_all", "quantization_tables", "play"]
    if args.task not in tasks:
        raise ValueError(f"Task {args.task} not supported, please choose between {tasks}")
    
    globals()[args.task](args)



if __name__ == '__main__':
    main()
