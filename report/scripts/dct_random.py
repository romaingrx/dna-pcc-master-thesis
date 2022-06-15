#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 June 07, 17:21:24
@last modified : 2022 June 07, 17:22:55
"""

import numpy as np
from scipy.fft import dct

r = np.random.uniform(0, 256, (8,8)).astype(np.uint8) - 128
d = dct(r)
