#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 May 22, 13:26:00
@last modified : 2022 May 24, 11:55:24
"""

from omegaconf import OmegaConf
from typing import Union, Generator, Iterable, List, Tuple, Dict, Any

class Namespace(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, data={}, *args, **kwargs):
        if kwargs.pop("_recursively", True):
            super().__init__(Namespace._map(data), *args, **kwargs)
        else:
            super().__init__(data, *args, **kwargs)

    @classmethod
    def _map(cls, data: Any) -> Union[Dict, Any]:
        """Recursively convert all dicts to Namespace"""
        if type(data) not in (dict, list, tuple):
            return data
        elif type(data) is list:
            return [cls._map(item) for item in data]
        elif type(data) is tuple:
            return tuple(cls._map(item) for item in data)
        return Namespace(
            {key: cls._map(value) for key, value in data.items()}, _recursively=False
        )

def omegaconf2namespace(config: OmegaConf, allow_missing:bool=False) -> Namespace:
    """
    Ensure that the configuration is valid and return an Namespace.
    """
    cfg_dict = OmegaConf.to_container(config, resolve=True)
    return dict2namespace(cfg_dict, allow_missing)


def dict2namespace(config: dict, allow_missing:bool=False, parent_keys:List[str]=[]) -> Namespace:
    """
    Ensure that the configuration is valid and return an Namespace.
    """
    namespace = Namespace(config)
    for key, value in config.items():
        if value == "???":
            raise ValueError(f"Missing value for `{'.'.join(parent_keys+[key])}`")
        elif isinstance(value, dict):
            namespace[key] = dict2namespace(value, allow_missing, parent_keys+[key])
    return namespace

def encode_wrapper(encode_fn):
    """Wrap the encode function to have the right shape for the encoder."""
    def inner_fn(self, x):
        """Reshape the input to have the right shape for the encoder."""
        # Input shape: (batch, b1, b2, b3, channels)
        assert (
                len(x.shape) == 5
                ), "x must be a 5D tensor (batch, b1, b2, b3, channels)"

        input_x = self.reshape_input_encode(x)
        res = encode_fn(self, input_x)
        return self.reshape_output_encode(res)
    return inner_fn

def decode_wrapper(decode_fn):
    """Wrap the decode function to have the right shape for the decoder."""
    def inner_fn(self, x, shape):
        """Reshape the input to have the right shape for the decoder."""
        # Output shape: (batch, b1, b2, b3, channels)
        assert len(shape) == 4, "shape must be a 3D tensor (b1, b2, b3, channels)"

        input_x = self.reshape_input_decoder(x, shape)
        res = decode_fn(self, input_x, shape)
        return self.reshape_output_decode(res, shape)
    return inner_fn
