#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 Mar 30, 21:59:48
@last modified : 2022 Mar 30, 22:49:31
"""

import requests


class MESA:
    __config = {
        "enabledUndesiredSeqs": [],
        "kmer_windowsize": "10",
        "gc_windowsize": "50",
        "gc_name": "Default Graph",
        "gc_error_prob": {
            "data": [
                {"x": 0, "y": 100},
                {"x": 30, "y": 100},
                {"x": 40, "y": 0},
                {"x": 60.17, "y": 0},
                {"x": 70, "y": 100},
                {"x": 100, "y": 100},
            ],
            "interpolation": True,
            "maxX": 100,
            "maxY": 100,
            "xRound": 2,
            "yRound": 2,
            "label": "Error Probability",
            "xLabel": "GC-Percentage",
        },
        "homopolymer_error_prob": {
            "data": [
                {"x": 0, "y": 0},
                {"x": 2, "y": 0},
                {"x": 4, "y": 0},
                {"x": 5, "y": 0},
                {"x": 6, "y": 0},
                {"x": 7, "y": 0},
                {"x": 8, "y": 100},
                {"x": 20, "y": 100},
            ],
            "interpolation": True,
            "label": "Error Probability",
            "maxX": 20,
            "maxY": 100,
            "xLabel": "Homopolymer length",
            "xRound": 0,
            "yRound": 2,
        },
        "homopolymer_name": "0-to-7",
        "kmer_error_prob": {
            "data": [
                {"x": 0, "y": 0},
                {"x": 3, "y": 0},
                {"x": 10, "y": 100},
                {"x": 40, "y": 100},
                {"x": 60, "y": 100},
                {"x": 100, "y": 100},
            ],
            "interpolation": True,
            "label": "Error Probability",
            "maxX": 20,
            "maxY": 100,
            "xLabel": "Kmer repeats",
            "xRound": 0,
            "yRound": 2,
        },
        "kmer_name": "0to3",
        "err_simulation_order": {
            "Sequencing": [
                {
                    "name": "Paired End",
                    "id": "36",
                    "cycles": 1,
                    "conf": {
                        "err_data": {
                            "deletion": 0.0018,
                            "insertion": 0.0011,
                            "mismatch": 0.79,
                            "raw_rate": 0.0032,
                        },
                        "err_attributes": {
                            "deletion": {
                                "pattern": {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
                                "position": {"random": 1},
                            },
                            "insertion": {
                                "pattern": {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
                                "position": {"random": 1},
                            },
                            "mismatch": {
                                "pattern": {
                                    "A": {"C": 0.25, "G": 0.5, "T": 0.25},
                                    "C": {"A": 0.25, "G": 0.5, "T": 0.25},
                                    "G": {"A": 0.25, "C": 0.25, "T": 0.5},
                                    "T": {"A": 0.25, "C": 0.25, "G": 0.5},
                                }
                            },
                        },
                        "type": "sequencing",
                    },
                }
            ],
            "Synthesis": [
                {
                    "name": "ErrASE",
                    "id": "3",
                    "cycles": 1,
                    "conf": {
                        "err_data": {
                            "deletion": 0.6,
                            "insertion": 0.2,
                            "mismatch": 0.2,
                            "raw_rate": 0.000025,
                        },
                        "err_attributes": {
                            "deletion": {
                                "pattern": {"A": 0.4, "C": 0.2, "G": 0.2, "T": 0.2},
                                "position": {"homopolymer": 0, "random": 1},
                            },
                            "insertion": {
                                "pattern": {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
                                "position": {"homopolymer": 0, "random": 1},
                            },
                            "mismatch": {"pattern": {}},
                        },
                        "type": "synthesis",
                    },
                }
            ],
            "Storage/PCR": [
                {
                    "name": "E coli",
                    "id": "4",
                    "cycles": "24",
                    "conf": {
                        "err_data": {
                            "deletion": 0.08,
                            "insertion": 0.08,
                            "mismatch": 0.84,
                            "raw_rate": 3.17e-7,
                        },
                        "err_attributes": {
                            "deletion": {
                                "pattern": {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
                                "position": {"homopolymer": 0, "random": 1},
                            },
                            "insertion": {
                                "pattern": {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
                                "position": {"homopolymer": 0, "random": 1},
                            },
                            "mismatch": {"pattern": {}},
                        },
                        "type": "storage",
                    },
                },
                {
                    "name": "Taq",
                    "id": "2",
                    "cycles": "30",
                    "conf": {
                        "err_data": {
                            "deletion": 0.01,
                            "insertion": 0,
                            "mismatch": 0.99,
                            "raw_rate": 0.000043,
                        },
                        "err_attributes": {
                            "deletion": {
                                "pattern": {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
                                "position": {"homopolymer": 0, "random": 1},
                            },
                            "insertion": {
                                "pattern": {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
                                "position": {"homopolymer": 0, "random": 1},
                            },
                            "mismatch": {
                                "pattern": {
                                    "A": {"C": 0.02, "G": 0.97, "T": 0.01},
                                    "C": {"A": 0, "G": 0, "T": 1},
                                    "G": {"A": 1, "C": 0, "T": 0},
                                    "T": {"A": 0.01, "C": 0.97, "G": 0.02},
                                }
                            },
                        },
                        "type": "pcr",
                    },
                },
            ],
        },
        "use_error_probs": False,
        "acgt_only": True,
        "random_seed": "",
        "do_max_expect": True,
        "temperature": "310.15",
        "send_mail": False,
        "email": "",
        "asHTML": False,
    }

    def __init__(self, host: str, key: str, port: int = 5000, secure: bool = False):
        self._host = host
        self.__config["key"] = key
        self._port = port
        self._secure = secure

    def _ensure_valid_sequence(self, sequence: str):
        assert set(sequence).issubset(
            set("ACGT")
        ), f'The sequence "{sequence}" is not valid since it is not containing only A, C, G or T'

    def _get_url(self, route: str) -> str:
        assert route.startswith("/"), "The route needs to start with /"
        return f'http{"s" if self._secure else ""}://{self._host}:{self._port}{route}'

    def post(self, route: str, data: dict) -> requests.Response:
        url = self._get_url(route)
        data = self.__config | data
        return requests.post(url, json=data)

    def simulate(self, sequence: str) -> tuple[bool, dict]:
        self._ensure_valid_sequence(sequence)
        data = {"sequence": sequence}
        r = self.post("/api/all", data)
        if r.status_code == 200:
            return True, r.json()[sequence]["res"]
        return False, r.json()


if __name__ == "__main__":
    simulator = MESA("127.0.0.1", "QZn8totJ-wl0631uShEY0-5CdWh-sW-WK98EXmhHoq8")
    succeed, response = simulator.simulate("AAGGTCTTCAGGTCAG")
    print(response)
    print(f'Has been modified? {response["sequence"] != response["modified_sequence"]}')
