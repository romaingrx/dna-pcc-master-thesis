"""Helper functions for .mat data"""
import _pickle as pickle
from scipy.io import loadmat
import gc
from functools import lru_cache

@lru_cache(maxsize=None)
def load_lut_matrix(string):
    """Loads matrix values saved in .mat file"""
    return loadmat(string)["lut"]

@lru_cache(maxsize=None)
def load_codebook_matrix(string):
    """Loads matrix values saved in .pkl file"""
    print("Loading codebook matrix... ", string)
    with open(string, "rb") as fd:
        gc.disable()
        arr = pickle.load(fd)
        gc.enable()
    return arr
