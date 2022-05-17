#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 May 13, 11:16:42
@last modified : 2022 May 13, 11:16:42
"""

import tensorflow as tf

class Residual_Block(tf.keras.layers.Layer):
    """Residual transform used in the Analysis and Synthesis transform"""

    def __init__(self, num_filters, name):
        super().__init__(name=name)
        self.block1 = tf.keras.layers.Conv3D(
            num_filters / 4, (3, 3, 3), padding="same", activation="relu"
        )

        self.block2 = tf.keras.layers.Conv3D(
            num_filters / 2, (3, 3, 3), padding="same", activation="relu"
        )

        self.block3 = tf.keras.layers.Conv3D(
            num_filters / 4, (1, 1, 1), padding="same", activation="relu"
        )

        self.block4 = tf.keras.layers.Conv3D(
            num_filters / 4, (3, 3, 3), padding="same", activation="relu"
        )

        self.block5 = tf.keras.layers.Conv3D(
            num_filters / 2, (1, 1, 1), padding="same", activation="relu"
        )

        self.concat = tf.keras.layers.Concatenate()
        self.add = tf.keras.layers.Add()

    def call(self, x):
        y1 = self.block1(x)
        y1 = self.block2(y1)

        y2 = self.block3(x)
        y2 = self.block4(y2)
        y2 = self.block5(y2)

        concat = self.concat([y1, y2])
        output = self.add([x, concat])
        return output


class AnalysisTransform(tf.keras.layers.Layer):
    """Analysis transform used to turn the input into its latent representation"""

    def __init__(self, num_filters, latent_depth):
        super().__init__(name="analysis")

        self.conv = tf.keras.layers.Conv3D(
            num_filters, (9, 9, 9), strides=(2, 2, 2), padding="same", activation="relu"
        )

        self.conv_int = tf.keras.layers.Conv3D(
            num_filters, (5, 5, 5), strides=(2, 2, 2), padding="same", activation="relu"
        )

        self.convout = tf.keras.layers.Conv3D(
            latent_depth,
            (5, 5, 5),
            strides=(2, 2, 2),
            padding="same",
            activation="linear",
        )

        self.res_block1 = Residual_Block(num_filters, name="block_1")
        self.res_block2 = Residual_Block(num_filters, name="block_2")

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.res_block1(x)
        x = self.conv_int(x)
        x = self.res_block2(x)
        x = self.convout(x)
        return x


class SynthesisTransform(tf.keras.layers.Layer):
    """Analysis transform used to turn the input into its latent representation"""

    def __init__(self, num_filters):
        super().__init__(name="synthesis")

        self.conv1 = tf.keras.layers.Conv3DTranspose(
            num_filters, (5, 5, 5), strides=(2, 2, 2), padding="same", activation="relu"
        )

        self.conv2 = tf.keras.layers.Conv3DTranspose(
            num_filters, (5, 5, 5), strides=(2, 2, 2), padding="same", activation="relu"
        )

        self.conv3 = tf.keras.layers.Conv3DTranspose(
            1, (9, 9, 9), strides=(2, 2, 2), padding="same", activation="sigmoid"
        )

        self.block1 = Residual_Block(num_filters, name="block_3")
        self.block2 = Residual_Block(num_filters, name="block_4")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.block1(x)
        x = self.conv2(x)
        x = self.block2(x)
        x = self.conv3(x)
        return x
