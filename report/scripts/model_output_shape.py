#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 June 07, 13:00:33
@last modified : 2022 June 07, 13:25:09
"""

import sys    
sys.path.append('../../compression/dna/codec')     
    
from layers import AnalysisTransform, SynthesisTransform    

import tensorflow as tf

num_filters = 64
latent_depth = 160

analysis = AnalysisTransform(num_filters, latent_depth)
synthesis = SynthesisTransform(num_filters)

for input_dims in range(64, 128):
    input_shape = [input_dims]*3 + [1]
    input_layer = tf.keras.layers.Input(shape=input_shape)
    latent_repr = analysis(input_layer)
    output_layer = synthesis(latent_repr)
    print(f'{input_dims} -> {latent_repr.shape[1:]}-> {output_layer.shape[1:]}')

