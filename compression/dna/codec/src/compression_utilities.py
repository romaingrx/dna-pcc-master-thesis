#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.spatial
import numpy as np
import tensorflow as tf
import sys


def bytes_to_nucleotides(bytestream):
    """
    Converts a bytestream into a nucleotide string.

    Parameters:
    bytestream: Bytestream to convert.

    Output: A nucleotide string.
    """

    bitstream = bin(int.from_bytes(bytestream, byteorder="little")).lstrip("0b")
    if len(bitstream) % 8 != 0:
        bitstream = (
            "0" * (8 - len(bitstream) % 8) + bitstream
        )  # Pad with zeros to make it divisible by 8.

    return "".join(
        [
            {"00": "A", "01": "C", "10": "G", "11": "T"}[bitstream[i : i + 2]]
            for i in range(0, len(bitstream), 2)
        ]
    )


def nucleotides_to_bytes(nucleotides, bits_type=8):
    """
    Converts a nucleotide string into a bytestream.

    Parameters:
    nucleotides: Nucleotide string to convert.

    Output: A bytestream.
    """
    bistream = "".join(
        [
            {"A": "00", "C": "01", "G": "10", "T": "11"}[nucleotide]
            for nucleotide in nucleotides
        ]
    )

    length_bytes = (len(bistream) % bits_type != 0) + len(bistream) // bits_type
    bytestream = int(bistream, 2).to_bytes(length_bytes, byteorder="little")
    return bytestream


def pack_tensor(
    threshold: tf.float32,
    oligo_length: tf.int32,
    y_shape: tf.constant,
    z_strings: tf.ragged.constant,
) -> str:
    """
    Pack the tensor of nucleotides into a single nucelotide string.

    Parameters:
        threshold: Threshold to transform the output of the NN into an occupancy map.
        y_shape: Shape of the analyis tensor (b1, b2, b3, latent_depth).
        z_strings: List of tensors of the latent DNA.

    Returns:
        A string of DNA nucleotides with structure:
        [threshold, y_shape, *z_length, *z_strings]

    Remarks:
        All lengths are encoded on 1 byte, so the maximum length is 255.
        It is possible to encode on several bytes if needed.
    """

    threshold_int = (np.round(threshold.numpy() * 100)).astype("uint8")
    threshold_string = bytes_to_nucleotides(bytes([threshold_int]))
    oligo_length_string = bytes_to_nucleotides(bytes([oligo_length]))
    y_shape_string = bytes_to_nucleotides(bytes(y_shape.numpy().tolist()))
    z_length_strings = bytes_to_nucleotides(
        bytes([len(z_string) for z_string in z_strings.numpy()])
    )

    return (
        threshold_string
        + oligo_length_string
        + y_shape_string
        + z_length_strings
        + "".join(tf.reshape(z_strings, (-1,)).numpy().astype(str))
    )


def unpack_tensor(nucelotidestream):
    """
    Unpack the tensor of nucleotides into a single nucelotide string.

    Parameters:
        nucelotidestream: Nucelotide string to unpack.

    Returns:
        A tuple of the threshold, y_shape, z_length, and z_strings.
    """

    seeker = 0

    threshold_string = nucelotidestream[seeker : seeker + 4]
    threshold_int = int.from_bytes(
        nucleotides_to_bytes(threshold_string), byteorder="little"
    )
    threshold = tf.constant(threshold_int / 100, dtype=tf.float32)
    seeker += 4

    oligo_length_string = nucelotidestream[seeker : seeker + 4]
    oligo_length = int.from_bytes(
        nucleotides_to_bytes(oligo_length_string), byteorder="little"
    )
    seeker += 4

    y_shape_string = nucelotidestream[seeker : seeker + 16]
    _, _, _, latent_depth = y_shape = np.frombuffer(
        nucleotides_to_bytes(y_shape_string), dtype=np.uint8
    )
    seeker += 16

    z_length_strings = nucelotidestream[seeker : seeker + latent_depth * 4]
    z_lengths = np.frombuffer(nucleotides_to_bytes(z_length_strings), dtype=np.uint8)
    seeker += latent_depth * 4

    z_strings = []
    for z_length in z_lengths:
        # The default oligo length is 200.
        z_string = [
            nucelotidestream[
                seeker + oligo_length * idx : seeker + oligo_length * (idx + 1)
            ]
            for idx in range(z_length)
        ]
        z_strings.append(z_string)
        seeker += z_length * oligo_length

    return (
        tf.constant(threshold, dtype=tf.float32),
        tf.constant(oligo_length, dtype=tf.int32),
        tf.constant(y_shape, dtype=tf.int32),
        tf.ragged.constant(z_strings, dtype=tf.string),
    )


def po2po(block1_pc, block2_pc):
    """
    Compute the point to point (D1) metric between two blocks.

    Parameters:
    block1_pc: First block to compare.
    block2_pc: Second block to compare.

    Output: The D1 metric between block1_pc and block2_pc.
    """

    # A -> B point to point metric.
    block1_tree = scipy.spatial.cKDTree(block1_pc)
    nearest_ref1 = block1_tree.query(block2_pc)
    po2po_ref1_mse = (nearest_ref1[0] ** 2).mean()

    # B -> A point to point metric.
    block2_tree = scipy.spatial.cKDTree(block2_pc)
    nearest_ref2 = block2_tree.query(block1_pc)
    po2po_ref2_mse = (nearest_ref2[0] ** 2).mean()

    # D1 is the max between the two above.
    po2po_mse = np.max((po2po_ref1_mse, po2po_ref2_mse))

    return po2po_mse


def compute_optimal_threshold(
    model, z_strings, y_shape, pc, delta_t=0.01, breakpt=50, verbose=1
):
    """
    Computes the optimal threshold used to convert the output of the
    neural network into an occupancy map.

    Parameters:
    tensors: tuple containing the shapes and string
             that the model.compress function outputs.
    delta_t: Space between two consecutive threshold
             during the grid search.
    breakpt: Number of thresholds to try without improvement
    verbose: Level of verbosity. Either 0 (no printing),
             1 (partial printing) or 2 (full printing)

    Output: The optimal threshold.
    """

    assert verbose in {
        0,
        1,
        2,
    }, "Verbose should be either 0(no printing), 1 (partial printing) or 2 (full printing)"
    # Decompress the latent tensor.
    x_hat = tf.squeeze(model.decompress(tf.expand_dims(z_strings, 0), y_shape))
    x_hat = x_hat.numpy()

    # Prepare parameters for search.
    num_not_improve = 0
    thresholds = tf.linspace(
        delta_t, 1, tf.cast(tf.math.round(1 / delta_t), dtype=tf.int64)
    )
    min_mse = 1e10
    best_threshold = tf.constant(0)

    for threshold in thresholds:

        # Locate values above current test threshold.
        pa = np.argwhere(x_hat > threshold).astype("float32")

        # Compute the associated D1 metric.
        mse = po2po(pc, pa)

        # Empty PC cause D1 metric to be NaN.
        if np.isnan(mse):

            # Try having only one point in the middle of the
            # block, if this solution is better than the one
            # found so far, an empty block is the best solution.
            mean_pt = np.round(np.mean(pc, axis=0))[np.newaxis, :]
            test_mse = po2po(pc, mean_pt)
            if verbose == 2:
                print(
                    f"The D1 error for the mean point is {test_mse}, against {min_mse} for the current best threshold."
                )

            # If the mean point is better than current threshold,
            # return an empty block (threshold = 1).
            if test_mse < min_mse:
                best_threshold = tf.constant(1)

            # If the adaptive threshold finds a threshold too
            # low, return the fixed threshold.
            if best_threshold.numpy() < 0.1:
                best_threshold = tf.constant(0.5)
            if verbose >= 1:
                print(f" Best threshold found: {best_threshold.numpy()}")
            return best_threshold

        # Update the current best threshold if necessary.
        if mse < min_mse:
            min_mse = mse
            best_threshold = threshold
            num_not_improve = 0
            if verbose == 2:
                print(
                    f"D1 mse value of {min_mse} found at t = {best_threshold.numpy()}"
                )
        else:
            num_not_improve += 1
            if verbose == 2:
                print(f"Not a better threshold mse = {mse} at t = {threshold.numpy()}")
            if num_not_improve == breakpt:
                return best_threshold


if __name__ == "__main__":
    _, _, _, latent_depth = y_shape = tf.constant([64, 64, 64, 160])
    oligo_length = np.random.randint(10, 255)
    z_strings = tf.ragged.constant(
        [
            [
                "".join(np.random.choice(["A", "C", "G", "T"], size=(oligo_length,)))
                for _ in range(np.random.randint(2, 10))
            ]
            for _ in range(latent_depth)
        ]
    )
    threshold = (
        tf.cast(tf.random.uniform((1,), 0, 100, dtype=tf.int32), tf.float32) / 100.0
    )[0]

    packed = pack_tensor(threshold, oligo_length, y_shape, z_strings)
    (
        unpacked_threshold,
        unpacked_oligo_length,
        unpacked_y_shape,
        unpacked_z_strings,
    ) = unpack_tensor(packed)

    assert unpacked_threshold == threshold
    assert unpacked_oligo_length == oligo_length
    assert (y_shape == unpacked_y_shape).numpy().all()
    assert (
        (z_strings.bounding_shape() == unpacked_z_strings.bounding_shape())
        .numpy()
        .all()
    )
    assert z_strings.to_list() == unpacked_z_strings.to_list()
    print("All tests passed.")
