#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 Mar 30, 21:59:48
@last modified : 2022 May 18, 18:09:12
"""

import requests
import numpy as np

import hydra
from omegaconf import OmegaConf, DictConfig

from functools import partial, lru_cache
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count

from typing import Union, Generator, Iterable, List, Tuple, Dict, Any


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, data, *args, **kwargs):
        if kwargs.pop("_recursively", True):
            super().__init__(AttrDict._map(data), *args, **kwargs)
        else:
            super().__init__(data, *args, **kwargs)

    @classmethod
    def _map(cls, data: Any) -> Union[Dict, Any]:
        """Recursively convert all dicts to AttrDict"""
        if type(data) not in (dict, list, tuple):
            return data
        elif type(data) is list:
            return [cls._map(item) for item in data]
        elif type(data) is tuple:
            return tuple(cls._map(item) for item in data)
        return AttrDict(
            {key: cls._map(value) for key, value in data.items()}, _recursively=False
        )


class MESAResponse:
    def __init__(self, response):
        self._response = response
        self._status_code = response.status_code

    @property
    @lru_cache(maxsize=1)
    def _json(self) -> AttrDict:
        """Lazy load json"""
        return AttrDict(self._response.json())

    @property
    @lru_cache(maxsize=1)
    def sequence(self) -> str:
        keys = list(self._json.keys())
        assert len(keys) == 1, f"Expected 1 key, got {len(keys)} keys ({keys})"
        return keys[0]

    @property
    @lru_cache(maxsize=1)
    def modified_sequence(self) -> str:
        return self._json[self.sequence].res.modified_sequence

    def is_ok(self) -> bool:
        return self._status_code == 200

    def __repr__(self) -> str:
        return (
            f"<MESAResponse status_code={self._status_code} sequence={self.sequence}>"
        )

    def __str__(self) -> str:
        return repr(self)


class MESASimulator:
    def __init__(self, config: DictConfig):
        self.__config = config

    def set_config(self, key: str, value: Any):
        self.__config.__setattr__(key, value)

    @property
    def n_workers(self) -> int:
        nb = self.__config.connection.n_workers
        return cpu_count() if nb == "all" else int(nb)

    def _ensure_valid_sequences(self, sequences: Union[List, Tuple]):
        for sequence in sequences:
            assert set(sequence).issubset(
                set("ACGT")
            ), f'The sequence "{sequence}" is not valid since it is not containing only A, C, G or T'

    def _get_url(self, route: str) -> str:
        assert route.startswith("/"), "The route needs to start with /"
        return f'http{"s" if self.__config.connection.secure else ""}://{self.__config.connection.host}:{self.__config.connection.port}{route}'

    def get_config(self, key: str = None) -> Dict:
        conf = self.__config
        if key is not None:
            conf = self.__config.__getattr__(key)
        return OmegaConf.to_object(conf)

    def post(
        self, data: Union[dict, Iterable[dict]], route: str
    ) -> Union[requests.Response, List[requests.Response]]:
        global post_worker

        def post_worker(data: dict, url: str, config: Dict) -> requests.Response:
            json = config | data
            return requests.post(url, json=json)

        url = self._get_url(route)
        config = self.get_config("post")

        if type(data) is dict:
            return post_worker(data, url, config)

        with ThreadPool(self.n_workers) as p:
            f = partial(post_worker, url=url, config=config)
            responses = p.map(f, data)

        return responses

    def simulate(self, sequences: Union[str, List, Tuple]) -> MESAResponse:
        sequences = sequences if type(sequences) in (list, tuple, np.ndarray) else [sequences]
        print(f"Simulating {sequences}")
        self._ensure_valid_sequences(sequences)
        data_sequences = [{"sequence": sequence} for sequence in sequences]
        responses = self.post(data_sequences, "/api/all")
        return (
            [MESAResponse(response) for response in responses]
            if len(sequences) > 1
            else MESAResponse(responses[0])
        )

@hydra.main(config_path="config/mesa", config_name="default.yaml")
def main(config: dict) -> None:
    global mesa, conf, y, sim_n, z
    conf = config
    mesa = MESASimulator(config)

    sim_n = n_dimensional(mesa.simulate)
    y = sim_n(sequences=z, from_arg="sequences")

    # status, res = mesa.simulate("ACGTACGTACGT")
    # print("Not modified sequence? ", res["modified_sequence"] == res["sequence"])
    # print(res)


if __name__ == "__main__":
    main()
