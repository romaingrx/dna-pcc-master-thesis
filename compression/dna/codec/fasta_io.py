#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 June 09, 17:48:35
@last modified : 2022 June 09, 18:07:41
"""

import re
from src.compression_utilities import unpack_tensor_single

def latent_representation_to_fasta(latent_representation, length):
    header_seeker = unpack_tensor_single(latent_representation, get_header_length=True)
    sequences = [latent_representation[i:i+length] for i in range(header_seeker, len(latent_representation), length)]
    fast_str = ""
    fast_str += "> header\n"
    fast_str += latent_representation[:header_seeker] + "\n"
    fast_str += "> oligo\n"
    fast_str += "\n> oligo\n".join(sequences)
    return fast_str

def get_value_from_fasta(key, fasta_str):
    values = re.findall(f">\s*{key}\s*\n([ACGT]+)\n", fasta_str)
    return values[0] if len(values) == 1 else values
