#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 May 05, 10:04:12
@last modified : 2022 May 05, 10:13:10
"""

import os
import tensorflow as tf
import tensorflow_compression as tfc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Residual_Block(tf.keras.layers.Layer):
    """Residual transform used in the Analysis and Synthesis transform"""

    def __init__(self, num_filters, name):
        super().__init__(name=name)
        self.block1 = tf.keras.layers.Conv3D(
                num_filters/4, (3, 3, 3), padding='same',
                activation='relu')

        self.block2 = tf.keras.layers.Conv3D(
                num_filters/2, (3, 3, 3), padding='same',
                activation='relu')

        self.block3 = tf.keras.layers.Conv3D(
                num_filters/4, (1, 1, 1), padding='same',
                activation='relu')

        self.block4 = tf.keras.layers.Conv3D(
                num_filters/4, (3, 3, 3), padding='same',
                activation='relu')

        self.block5 = tf.keras.layers.Conv3D(
                num_filters/2, (1, 1, 1), padding='same',
                activation='relu')

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
        super().__init__(name='analysis')

        self.conv = tf.keras.layers.Conv3D(
                num_filters, (9, 9, 9), strides=(2, 2, 2), padding='same',
                activation='relu')

        self.conv_int = tf.keras.layers.Conv3D(
                num_filters, (5, 5, 5), strides=(2, 2, 2), padding='same',
                activation='relu')

        self.convout = tf.keras.layers.Conv3D(
                latent_depth, (5, 5, 5), strides=(2, 2, 2), padding='same',
                activation='linear')

        self.res_block1 = Residual_Block(num_filters, name='block_1')
        self.res_block2 = Residual_Block(num_filters, name='block_2')

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
        super().__init__(name='synthesis')

        self.conv1 = tf.keras.layers.Conv3DTranspose(
                num_filters, (5, 5, 5), strides=(2, 2, 2), padding='same',
                activation='relu')

        self.conv2 = tf.keras.layers.Conv3DTranspose(
                num_filters, (5, 5, 5), strides=(2, 2, 2), padding='same',
                activation='relu')

        self.conv3 = tf.keras.layers.Conv3DTranspose(
                1, (9, 9, 9), strides=(2, 2, 2), padding='same',
                activation='sigmoid')

        self.block1 = Residual_Block(num_filters, name='block_3')
        self.block2 = Residual_Block(num_filters, name='block_4')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.block1(x)
        x = self.conv2(x)
        x = self.block2(x)
        x = self.conv3(x)
        return x


class HyperAnalysisTransform(tf.keras.layers.Layer):
    """Analysis transform for the entropy model's parameters."""

    def __init__(self, num_filters, hyperprior_depth):
        super().__init__(name='hyper_analysis')

        self.conv1 = tf.keras.layers.Conv3D(
                num_filters, (3, 3, 3), padding='same',
                activation='relu')

        self.conv2 = tf.keras.layers.Conv3D(
                num_filters, (3, 3, 3), strides=(2, 2, 2), padding='same',
                activation='relu')

        self.conv3 = tf.keras.layers.Conv3D(
                hyperprior_depth, (1, 1, 1), padding='same',
                activation='linear', use_bias=False)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class HyperSynthesisTransform(tf.keras.layers.Layer):
    """Synthesis transform for the entropy model's parameters."""

    def __init__(self, num_filters):
        super().__init__(name='hyper_synthesis')

        self.convt1 = tf.keras.layers.Conv3DTranspose(
                num_filters, (1, 1, 1), padding='same',
                activation='relu')

        self.convt2 = tf.keras.layers.Conv3DTranspose(
                num_filters, (3, 3, 3), strides=(2, 2, 2), padding='same',
                activation='relu')

        self.convt3 = tf.keras.layers.Conv3DTranspose(
                num_filters, (3, 3, 3), padding='same',
                activation='relu')

    def call(self, x):
        x = self.convt1(x)
        x = self.convt2(x)
        x = self.convt3(x)
        return x


class SliceTransform(tf.keras.layers.Layer):
    """Transform for channel-conditional params and latent residual prediction."""

    def __init__(self, num_filters, latent_depth, num_slices):
        super().__init__(name='slice_transform')

        # Note that the number of channels in the output tensor must match the
        # size of the corresponding slice. If we have 10 slices and a bottleneck
        # with 320 channels, the output is 320 / 10 = 32 channels.
        slice_depth = latent_depth // num_slices
        if slice_depth * num_slices != latent_depth:
            raise ValueError('Slices do not evenly divide latent depth (%d / %d)' % (
                latent_depth, num_slices))

        self.transform = tf.keras.Sequential([
            tf.keras.layers.Conv3D(
                num_filters, (3, 3, 3), padding='same',
                activation='relu'),
            tf.keras.layers.Conv3D(
                num_filters, (3, 3, 3), padding='same',
                activation='relu'),
            tf.keras.layers.Conv3D(
                slice_depth, (3, 3, 3), padding='same',
                activation='linear'),
            ])

    def call(self, tensor):
        return self.transform(tensor)


class CompressionModel(tf.keras.Model):
    """Main model class."""

    def __init__(self, lmbda, alpha, num_filters,
            latent_depth, hyperprior_depth,
            num_slices, max_support_slices,
            num_scales, scale_min, scale_max):
        super().__init__()
        self.lmbda = lmbda
        self.alpha = alpha
        self.latent_depth = latent_depth
        self.num_filters = num_filters
        self.num_scales = num_scales
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices

        offset = tf.math.log(scale_min)
        factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
                num_scales - 1.)
        self.scale_fn = lambda i: tf.math.exp(offset + factor * i)

        self.analysis_transform = AnalysisTransform(num_filters, latent_depth)
        self.synthesis_transform = SynthesisTransform(num_filters)

        self.hyper_analysis_transform = HyperAnalysisTransform(
                num_filters, hyperprior_depth)
        self.hyper_synthesis_mean_transform = HyperSynthesisTransform(
                num_filters)
        self.hyper_synthesis_scale_transform = HyperSynthesisTransform(
                num_filters)

        self.cc_mean_transforms = [
                SliceTransform(num_filters, latent_depth, num_slices) for _ in range(num_slices)]
        self.cc_scale_transforms = [
                SliceTransform(num_filters, latent_depth, num_slices) for _ in range(num_slices)]
        self.lrp_transforms = [
                SliceTransform(num_filters, latent_depth, num_slices) for _ in range(num_slices)]

        self.hyperprior = tfc.NoisyDeepFactorized(
                batch_shape=[hyperprior_depth])

        self.build((None, None, None, None, 1))

        # The call signature of decompress() depends on the number of slices, so we
        # need to compile the function dynamically.
        self.decompress = tf.function(
                input_signature=3 * [tf.TensorSpec(shape=(3,), dtype=tf.int32)] +
                (num_slices + 1) * [tf.TensorSpec(shape=(1,), dtype=tf.string)]
                )(self.decompress)

    def call(self, x, training):
        """Computes rate and distortion losses."""

        geo_x = x[:, :, :, :, 0]

        num_voxels = tf.cast(tf.size(geo_x), tf.float32)
        num_occupied_voxels = tf.reduce_sum(geo_x)

        # Build the encoder (analysis) half of the hierarchical autoencoder.
        y = self.analysis_transform(x)
        y_shape = tf.shape(y)[1:-1]

        # TODO: add the quantization transform here.

        # Build the encoder (analysis) half of the hyper-prior.
        z = self.hyper_analysis_transform(y)

        # Build the entropy model for the hyperprior (z).
        em_z = tfc.ContinuousBatchedEntropyModel(
                self.hyperprior, coding_rank=4, compression=False)

        # When training, z_bpp is based on the noisy version of z (z_tilde).
        _, z_bits = em_z(z, training=training)
        z_bpp = tf.reduce_sum(z_bits) / num_occupied_voxels

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_hat = em_z.quantize(z)

        # Build the decoder (synthesis) half of the hyper-prior.
        latent_scales = self.hyper_synthesis_scale_transform(z_hat)
        latent_means = self.hyper_synthesis_mean_transform(z_hat)

        # Build a conditional entropy model for the slices.
        em_y = tfc.LocationScaleIndexedEntropyModel(
                tfc.NoisyNormal, num_scales=self.num_scales, scale_fn=self.scale_fn,
                coding_rank=4, compression=False)

        # En/Decode each slice conditioned on hyperprior and previous slices.
        y_slices = tf.split(y, self.num_slices, axis=-1)
        y_hat_slices = []
        y_bpps = []
        for slice_index, y_slice in enumerate(y_slices):

            # Model may condition on only a subset of previous slices.
            support_slices = (y_hat_slices if self.max_support_slices < 0 else
                    y_hat_slices[:self.max_support_slices])

            # Predict mu and sigma for the current slice.
            mean_support = tf.concat([latent_means] + support_slices, axis=-1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :y_shape[0], :y_shape[1], :]

            # Note that in this implementation, `sigma` represents scale indices,
            # not actual scale values.
            scale_support = tf.concat(
                    [latent_scales] + support_slices, axis=-1)
            sigma = self.cc_scale_transforms[slice_index](scale_support)
            sigma = sigma[:, :y_shape[0], :y_shape[1], :]

            _, slice_bits = em_y(y_slice, sigma, loc=mu, training=training)
            slice_bpp = tf.reduce_sum(slice_bits) / num_occupied_voxels
            y_bpps.append(slice_bpp)

            # For the synthesis transform, use rounding. Note that quantize()
            # overrides the gradient to create a straight-through estimator.
            y_hat_slice = em_y.quantize(y_slice, sigma, loc=mu)

            # Add latent residual prediction (LRP).
            lrp_support = tf.concat([mean_support, y_hat_slice], axis=-1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * tf.math.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        # Merge slices and generate the bloc reconstruction.
        y_hat = tf.concat(y_hat_slices, axis=-1)
        x_hat = self.synthesis_transform(y_hat)

        # Total bpp is sum of bpp from hyperprior and all slices.
        total_bpp = tf.add_n(y_bpps + [z_bpp])
        y_bpp = tf.add_n(y_bpps)
        z_bpp = tf.add_n([z_bpp])

        # Compute the focal loss and/or color loss across pixels.
        # Don't clip or round pixel values while training.
        fcl = focal_loss(x, x_hat, gamma=2, alpha=self.alpha) / num_voxels

        # Calculate and return the rate-distortion loss.
        loss = total_bpp + self.lmbda * fcl

        return loss, y_bpp, z_bpp, total_bpp, fcl

    def train_step(self, x):
        with tf.GradientTape(persistent=True) as tape:
            # Compute the loss under the gradient tape to prepare for
            # gradient computation.
            loss, bpp1, bpp2, bpp, fcl = self(x, training=True)

        # Gather the trainable variables.
        variables = self.trainable_variables
        prior_var = self.hyperprior.trainable_variables

        # Compute the gradients w.r.t the trainable variables.
        gradients = tape.gradient(loss, variables)
        grad_prior = tape.gradient(loss, prior_var)

        # Apply the gradient to update the variables.
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.optimizer.apply_gradients(zip(grad_prior, prior_var))

        # Update the values displayed during training.
        self.loss.update_state(loss)
        self.bpp1.update_state(bpp1)
        self.bpp2.update_state(bpp2)
        self.bpp.update_state(bpp)
        self.fcl.update_state(self.lmbda*fcl)

        # Return the appropriate informations to be displayed on screen.
        return {m.name: m.result() for m in [self.loss, self.bpp1, self.bpp2, self.bpp, self.fcl]}

    def test_step(self, x):
        loss, bpp1, bpp2, bpp, fcl = self(x, training=False)

        # Update the values displayed during validation.
        self.loss.update_state(loss)
        self.bpp1.update_state(bpp1)
        self.bpp2.update_state(bpp2)
        self.bpp.update_state(bpp)
        self.fcl.update_state(self.lmbda*fcl)

        # Return the appropriate informations to be displayed on screen.
        return {m.name: m.result() for m in [self.loss, self.bpp1, self.bpp2, self.bpp, self.fcl]}

    def predict_step(self, x):
        raise NotImplementedError('Prediction API is not supported.')

    def compile(self, **kwargs):
        super().compile(
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                **kwargs,
                )
        self.loss = tf.keras.metrics.Mean(name='loss')
        self.bpp1 = tf.keras.metrics.Mean(name='direct_bpp')
        self.bpp2 = tf.keras.metrics.Mean(name='side_bpp')
        self.bpp = tf.keras.metrics.Mean(name='bpov')
        self.fcl = tf.keras.metrics.Mean(name='focal_loss')

    def fit(self, *args, **kwargs):
        retval = super().fit(*args, **kwargs)

        # After training, fix range coding tables.
        self.em_z = tfc.ContinuousBatchedEntropyModel(
                self.hyperprior, coding_rank=4, compression=True)

        self.em_y = tfc.LocationScaleIndexedEntropyModel(
                tfc.NoisyNormal, num_scales=self.num_scales, scale_fn=self.scale_fn,
                coding_rank=4, compression=True)
        return retval

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
        ])
    def compress(self, x):
        """Compresses a block."""

        # Add batch dimension and cast to float.
        x = tf.expand_dims(x, 0)

        y_strings = []
        x_shape = tf.shape(x)[1:-1]

        # Build the encoder (analysis) half of the hierarchical autoencoder.
        y = self.analysis_transform(x)
        y_shape = tf.shape(y)[1:-1]

        # Build the encoder (analysis) half of the hyper-prior.
        z = self.hyper_analysis_transform(y)
        z_shape = tf.shape(z)[1:-1]

        # Compress the output of the Hyper-Analysis to pass it
        # in the bistream.
        z_string = self.em_z.compress(z)
        z_hat = self.em_z.decompress(z_string, z_shape)

        # Build the decoder (synthesis) half of the hyper-prior.
        latent_scales = self.hyper_synthesis_scale_transform(z_hat)
        latent_means = self.hyper_synthesis_mean_transform(z_hat)

        # En/Decode each slice conditioned on hyperprior and previous slices.
        y_slices = tf.split(y, self.num_slices, axis=-1)
        y_hat_slices = []
        for slice_index, y_slice in enumerate(y_slices):

            # Model may condition on only a subset of previous slices.
            support_slices = (y_hat_slices if self.max_support_slices < 0 else
                    y_hat_slices[:self.max_support_slices])

            # Predict mu and sigma for the current slice.
            mean_support = tf.concat([latent_means] + support_slices, axis=-1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :y_shape[0], :y_shape[1], :]

            # Note that in this implementation, `sigma` represents scale indices,
            # not actual scale values.
            scale_support = tf.concat(
                    [latent_scales] + support_slices, axis=-1)
            sigma = self.cc_scale_transforms[slice_index](scale_support)
            sigma = sigma[:, :y_shape[0], :y_shape[1], :]

            slice_string = self.em_y.compress(y_slice, sigma, mu)
            y_strings.append(slice_string)
            y_hat_slice = self.em_y.decompress(slice_string, sigma, mu)

            # Add latent residual prediction (LRP).
            lrp_support = tf.concat([mean_support, y_hat_slice], axis=-1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * tf.math.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        return (x_shape, y_shape, z_shape, z_string) + tuple(y_strings)

    def decompress(self, x_shape, y_shape, z_shape, z_string, *y_strings):
        """Decompresses a block."""

        assert len(y_strings) == self.num_slices

        # Recover the entropy parameters.
        z_hat = self.em_z.decompress(z_string, z_shape)

        # Build the decoder (synthesis) half of the hierarchical autoencoder.
        latent_scales = self.hyper_synthesis_scale_transform(z_hat)
        latent_means = self.hyper_synthesis_mean_transform(z_hat)

        # En/Decode each slice conditioned on hyperprior and previous slices.
        y_hat_slices = []
        for slice_index, y_string in enumerate(y_strings):
            # Model may condition on only a subset of previous slices.
            support_slices = (y_hat_slices if self.max_support_slices < 0 else
                    y_hat_slices[:self.max_support_slices])

            # Predict mu and sigma for the current slice.
            mean_support = tf.concat([latent_means] + support_slices, axis=-1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :y_shape[0], :y_shape[1], :]

            # Note that in this implementation, `sigma` represents scale indices,
            # not actual scale values.
            scale_support = tf.concat(
                    [latent_scales] + support_slices, axis=-1)
            sigma = self.cc_scale_transforms[slice_index](scale_support)
            sigma = sigma[:, :y_shape[0], :y_shape[1], :]

            y_hat_slice = self.em_y.decompress(y_string, sigma, loc=mu)

            # Add latent residual prediction (LRP).
            lrp_support = tf.concat([mean_support, y_hat_slice], axis=-1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * tf.math.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        # Merge slices and generate the image reconstruction.
        y_hat = tf.concat(y_hat_slices, axis=-1)
        x_hat = self.synthesis_transform(y_hat)

        # Remove batch dimension, and crop away any extraneous padding.
        x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

        # The transformation of x_hat into an occupancy map is done at a
        # later stage to allow for the computation of the adaptive threshold.
        return x_hat
