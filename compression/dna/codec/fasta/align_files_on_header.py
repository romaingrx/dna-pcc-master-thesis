#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 June 09, 17:28:55
@last modified : 2022 June 09, 22:48:38
"""

import re
from os import system
from glob import glob

def extract_n_first(path, n):
    """
    Extract infos from a file
    """
    with open(path, 'r') as f:
        data = f.read()
    return tuple(re.findall(r'\n([ACGT]+)\n', data)[:n])

n = 5
input_files = glob('in/*')
output_files = glob('raw_fastq/*')

input_infos = [extract_n_first(f, n) for f in input_files]
output_infos = [extract_n_first(f, n) for f in output_files]

os.mkdir('simulated', exist_ok=True)

already_aligned = dict()
for in_name, in_info in zip(input_files, input_infos):
    for out_name, out_info in zip(output_files, output_infos):
        if in_info == out_info:
            if in_info in already_aligned:
                print(f"{in_name} -> {out_name}")
                print(f'Already aligned {in_info} with {already_aligned[in_info]}')
                continue

            already_aligned[in_info] = out_name
            system(f'cp "{out_name}" "{in_name.replace("in", "simulated")}"')
