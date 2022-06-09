#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 June 09, 17:28:55
@last modified : 2022 June 09, 22:07:45
"""

from os import system
from glob import glob

def extract_header(path):
    """
    Extract headers from a file
    """
    with open(path, 'r') as f:
        lines = f.readlines()
        return lines[1].strip()

input_files = glob('in/*')
output_files = glob('raw_fastq/*')

input_headers = [extract_header(f) for f in input_files]
output_headers = [extract_header(f) for f in output_files]


similitude = 0
for in_name, in_header in zip(input_files, input_headers):
    for out_name, out_header in zip(output_files, output_headers):
        if in_header == out_header:
            similitude += 1
            system(f'cp "{out_name}" "{in_name.replace("in", "aligned")}"')
print(f"Found {similitude} similitudes")


