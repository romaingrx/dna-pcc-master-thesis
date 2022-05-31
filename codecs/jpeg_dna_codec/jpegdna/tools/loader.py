"""Helper functions for .mat data"""
import _pickle as pickle
from scipy.io import loadmat
from functools import lru_cache
import gc

@lru_cache(maxsize=None)
def load_lut_matrix(string):
    """Loads matrix values saved in .mat file"""
    return loadmat(string)["lut"]

@lru_cache(maxsize=None)
def load_codebook_matrix(string):
    """Loads matrix values saved in .pkl file"""
    print("Loading codebook matrix... ", string)
    with open(string, "rb") as f:
        gc.disable()
        arr = pickle.load(f)
        gc.enable()
    return arr
