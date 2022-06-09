#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 Mar 30, 21:59:48
@last modified : 2022 June 09, 21:41:49
"""

import os
import re
import requests
import numpy as np
from bs4 import BeautifulSoup

import hydra
from omegaconf import OmegaConf, DictConfig

from functools import partial, lru_cache
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from tqdm import tqdm

from typing import Union, Generator, Iterable, List, Tuple, Dict, Any

from utils import n_dimensional
from helpers import Namespace
from benchmarks import load_io_files
from fasta_io import get_value_from_fasta


class MESAResponse:
    def __init__(self, response):
        self._response = response
        self._status_code = response.status_code
        self.__sequence_res = len(response.json().keys()) == 1

    @property
    @lru_cache(maxsize=1)
    def json(self) -> Namespace:
        """Lazy load json"""
        return Namespace(self._response.json())

    @property
    @lru_cache(maxsize=1)
    def sequence(self) -> str:
        keys = list(self.json.keys())
        if len(keys) == 1:
            return keys[0]
        elif list(keys) == ["query", "res", "uuid"]:
            return list(self.json["res"].keys())[0]

    @property
    @lru_cache(maxsize=1)
    def modified_sequence(self) -> str:
        if self.__sequence_res:
            return self.json[self.sequence].res.modified_sequence
        html_modified =  self.json["res"][self.sequence]["modify"]
        return BeautifulSoup(html_modified, "html.parser").text

    def is_ok(self) -> bool:
        return self._status_code == 200

    def __repr__(self) -> str:
        return (
            f"<MESAResponse status_code={self._status_code} sequence={self.sequence}>"
        )

    def __str__(self) -> str:
        return repr(self)


class MESASimulator:
    __headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:101.0) Gecko/20100101 Firefox/101.0",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Content-Type": "application/json;charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Length": "3026",
        "Origin": "http://localhost:5001",
        "DNT": "1",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Sec-GPC": "1",
        }

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
        s = ""
        s += "http"
        s += "s" if self.__config.connection.secure else ""
        s += "://"
        s += self.__config.connection.host
        s += ":" + str(self.__config.connection.port) if self.__config.connection.port else ""
        s += route
        return s

    def get_config(self, key: str = None) -> Dict:
        conf = self.__config
        if key is not None:
            conf = self.__config.__getattr__(key)
        return OmegaConf.to_object(conf)


    def post(
        self, data: Union[dict, Iterable[dict]], url: str, config
    ) -> Union[requests.Response, List[requests.Response]]:
        global post_worker

        def post_worker(data: dict, url: str, config: Dict, retry:int=0) -> requests.Response:
            json = {**config, **data}
            r = requests.post(url, json=json)
            if r.status_code != 200:
                if retry >= 3:
                    raise Exception(r.text)
                return post_worker(data, url, config, retry=retry+1)
            return r

        if type(data) is dict:
            return post_worker(data, url, config)

        with ThreadPool(self.n_workers) as p:
            f = partial(post_worker, url=url, config=config)
            responses = list(tqdm(p.imap(f, data), total=len(data), desc="Retrieving sequences from the web server"))

        return responses

    def simulate(self, sequences: Union[str, List, Tuple]) -> MESAResponse:
        sequences = sequences if type(sequences) in (list, tuple, np.ndarray) else [sequences]
        self._ensure_valid_sequences(sequences)
        data_sequences = [{"sequence": sequence} for sequence in sequences]
        responses = self.post(data_sequences, self._get_url("/api/all"), self.get_config("post"))
        return (
            [MESAResponse(response) for response in responses]
            if len(sequences) > 1
            else MESAResponse(responses[0])
        )

    def from_uuid(self, uuids: str) -> MESAResponse:
        data_sequences = [{"uuid": uuid} for uuid in uuids[:10]]
        url = self._get_url("/api/all")
        config = {"key":self.get_config("post")["key"], "asHTML":False}

        # Retrieve all modified sequences
        responses = self.post(data_sequences, url, config)

        return (
            [MESAResponse(response) for response in responses]
            if len(data_sequences) > 1
            else MESAResponse(responses[0])
        )

    def from_fastq(self, fastq: str) :
        uuids = re.findall("uuid=(.*)\n", fastq)
        return self.from_uuid(uuids)

    def simulate_fasta(self, sequences: Union[str, List, Tuple]) -> str:
        sequences = sequences if type(sequences) in (list, tuple, np.ndarray) else [sequences]
        self._ensure_valid_sequences(sequences)
        data = {"sequence_list": sequences}
        response = self.post(data, self._get_url("/api/fasta_all"), self.get_config("post"))
        return response

    @classmethod
    def concatenate(cls, sequences: Union[str, List, Tuple]) -> str:
        sequences = sequences if type(sequences) in (list, tuple, np.ndarray) else [sequences]
        return "".join([r.modified_sequence for r in sequences])

def get_modified_sequence_from_fastq(mesa, args):
    global files, all_sequences
    files = load_io_files(args.fasta, return_names=True, exception=["out"])
    all_sequences = {}
    for in_fasta, simulated_fasta, name in files:
        results = mesa.from_fastq(simulated_fasta)
        sequence = get_value_from_fasta('header', in_fasta) + mesa.concatenate(results[1:])
        all_sequences[name] = sequence
        with open(os.path.join(args.fasta.io.out, f"{name}.dna"), "w") as f:
            f.write(sequence)
    return all_sequences





@hydra.main(config_path="config/simulator", config_name="default.yaml", version_base="1.2")
def main(config: dict) -> None:
    global mesa, conf, y, sim_n, z, res, fastq, res
    conf = config
    mesa = MESASimulator(config)

    res = get_modified_sequence_from_fastq(mesa, config)


if __name__ == "__main__":
    main()
