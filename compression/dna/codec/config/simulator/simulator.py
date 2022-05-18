#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 Mar 30, 21:59:48
@last modified : 2022 Apr 04, 17:15:34
"""

import hydra
from omegaconf import OmegaConf, DictConfig

import requests

class MESA:
    def __init__(self, config: DictConfig):
        self.__config = config 

    def _ensure_valid_sequence(self, sequence: str):
        assert set(sequence).issubset(
            set("ACGT")
        ), f'The sequence "{sequence}" is not valid since it is not containing only A, C, G or T'

    def _get_url(self, route: str) -> str:
        assert route.startswith("/"), "The route needs to start with /"
        return f'http{"s" if self.__config.connection.secure else ""}://{self.__config.connection.host}:{self.__config.connection.port}{route}'

    def get_config(self, key:str=None) -> dict:
        conf = self.__config
        if key is not None:
            conf = self.__config.__getattr__(key)
        return OmegaConf.to_object(conf)

    def post(self, route: str, **data: dict) -> requests.Response:
        url = self._get_url(route)
        data = self.get_config("post") | data
        print(data)
        return requests.post(url, json=data)

    def simulate(self, sequence: str) -> tuple[bool, dict]:
        self._ensure_valid_sequence(sequence)
        r = self.post("/api/all", sequence=sequence)
        if r.status_code == 200:
            return True, r.json()[sequence]["res"]
        return False, r.json()

@hydra.main(config_path="config", config_name="default.yaml")
def main(config: dict) -> None:
    mesa = MESA(config)
    status, res = mesa.simulate("ACGTACGTACGT")
    print("Not modified sequence? ", res["modified_sequence"] == res["sequence"])
    print(res)

if __name__ == "__main__":
    main()
