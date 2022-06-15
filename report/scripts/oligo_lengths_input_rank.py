#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 June 07, 14:55:31
@last modified : 2022 June 07, 15:25:24
"""

import numpy as np
from tqdm import tqdm
from jpegdna.codecs import JpegDNA

block_size = 8
latent_depth = 160
block_shape = (block_size, block_size, block_size)

x1 = np.ones((block_size, block_size**2) + (latent_depth,))
x2 = np.ones((block_size ** 3,) + (latent_depth,))
x1 = np.transpose(x1, (2, 0, 1))

codec = JpegDNA(1)
# y1 = [[codec.encode(xii, "from_img", apply_dct=False) for xii in xi] for xi in tqdm(x1, desc=f"Encode x1 {x1.shape}")]
y1 = [codec.encode(xi, "from_img", apply_dct=False) for xi in tqdm(x1, desc=f"Encode x1 {x1.shape}")]
y2 = codec.encode(x2, "from_img", apply_dct=False)

print(f"Number of oligos for shape {x1.shape}: {len(np.reshape(y1, (-1,)))}")
print(f"Number of oligos for shape {x2.shape}: {len(np.reshape(y2, (-1,)))}")
