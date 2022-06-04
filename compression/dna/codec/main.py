# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 May 10, 11:15:58
@last modified : 2022 June 04, 15:34:48
"""

from functools import partial
from itertools import product
from tqdm import tqdm
import hydra
import logging
import numpy as np
import os
import tensorflow as tf
import tensorflow_compression as tfc
from jpegdna.codecs import JpegDNA, JPEGDNAGray

from ray.util.multiprocessing import Pool
# from multiprocessing import Pool


from helpers import Namespace, omegaconf2namespace, encode_wrapper, decode_wrapper
from layers import AnalysisTransform, SynthesisTransform
from src import pc_io
from src.compression_utilities import (
        pack_tensor_multi,
        unpack_tensor_multi,
        pack_tensor_single,
        unpack_tensor_single,
        compute_optimal_threshold,
        )
from src.focal_loss import focal_loss
from utils import (
        pc_dir_to_ds,
        number_of_nucleotides,
        )
from quantizer import quantize, dequantize, reduce_batch
from dna_io import save_all_intermediate_results, save_oligos

logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_codec(name:str, alpha:float):
    """Parse the codec name and the alpha value."""

    mapping = {
            "BatchSingleChannelJpegDNA" : (partial(BatchSingleChannelJpegDNA, alpha=alpha), pack_tensor_single, unpack_tensor_single),
            "BatchMultiChannelsJpegDNA" : (partial(BatchMultiChannelsJpegDNA, alpha=alpha), pack_tensor_multi, unpack_tensor_multi)
            }


    try:
        cls, pack, unpack = mapping[name]
        return cls(), pack, unpack
    except KeyError:
        raise ValueError(f"Unknown codec: {name}, please choose between {list(mapping.keys())}")


class BatchSingleChannelJpegDNA:
    oligo_length = 200  # Weird but impossible to adapt in Xavier's code

    def __init__(self, alpha):
        self._alpha = alpha
        self.num_workers = None
        self.gammas = None

    @property
    def num_workers(self):
        return self.__num_workers

    @num_workers.setter
    def num_workers(self, value):
        if value == "all" or value is None:
            self.__num_workers = os.cpu_count()
        else:
            if not isinstance(value, int):
                raise ValueError("num_workers must be an integer or 'all'")
            if value < 1:
                raise ValueError("num_workers must be greater than 0")
            self.__num_workers = value

    @property
    def gammas(self):
        return self.__gammas

    @gammas.setter
    def gammas(self, value):
        if value is None:
            self.__gammas = None
        else:
            if np.isscalar(value):
                if value <= 0:
                    raise ValueError("gammas must be greater than 0")
                self.__gammas = value * np.ones((8, 8))
            elif isinstance(value, np.ndarray):
                if value.shape != (8, 8):
                    raise ValueError("gammas must be a 2D array of shape (8, 8)")
                if not np.all(value > 0):
                    raise ValueError("gammas must be greater than 0")
                self.__gammas = value
            else:
                raise ValueError("gammas must be a scalar or a 2D array")
            JPEGDNAGray.GAMMAS = self.__gammas
            JPEGDNAGray.GAMMAS_CHROMA = self.__gammas


    def reshape_input_encode(self, x):
        """Reshape the input to have the right shape for the encoder."""
        return tf.reshape(x, (-1, x.shape[1] * x.shape[2] * x.shape[3], x.shape[4]))

    def reshape_output_encode(self, x):
        """Reshape the output of the encoder"""
        return x

    def reshape_input_decoder(self, x, shape):
        """Reshape the output of the decoder"""
        return x

    def reshape_output_decode(self, x, shape):
        """Reshape the input to have the right shape for the decoder."""
        return tf.reshape(x, [-1, *shape])

    @encode_wrapper
    def encode_batch(self, x, apply_dct=False):
        """Encode a batch of images with several channels into a list of oligos."""
        # Input shape: (batch, b1*b2*b3, channels)
        # Output shape: (batch, nb_oligos)
        assert (
                len(x.shape) == 3
                ), "x must be a 3D tensor (batch, height, width)"

        global _encode_worker

        def _encode_worker(x, alpha, apply_dct, gammas):
            """Encode a 2 dimensional array into several oligos."""
            # set the context
            if gammas is not None:
                JPEGDNAGray.GAMMAS = gammas
                JPEGDNAGray.GAMMAS_CHROMA = gammas

            return JpegDNA(alpha).encode(x.numpy(), "from_img", apply_dct=apply_dct)

        f = partial(_encode_worker, alpha=self._alpha, apply_dct=apply_dct, gammas=self.gammas)
        # y = [f(e) for e in x]
        with Pool(self.num_workers) as p:
            y = list(p.map(f, x))

        return tf.ragged.constant(y, dtype=tf.string)

    @decode_wrapper
    def decode_batch(self, x, shape, apply_dct=False):
        """Decode a batch of oligos into a batch of images with several channels."""
        # Input shape: (batch, nb_oligos)
        # Output shape: (batch, b1, b2, b3, channels)
        assert len(x.shape) == 2, "x must be a 2D tensor (batch, nb_oligos)"

        global _decode_worker

        def _decode_worker(oligos, alpha, apply_dct):
            """Decode a list of oligos into a 2 dimensional array."""
            return JpegDNA(alpha).decode(oligos.numpy().astype(str), apply_dct=apply_dct)

        f = partial(_decode_worker, alpha=self._alpha, apply_dct=apply_dct)
        # y = [f(e) for e in x]
        with Pool(self.num_workers) as p:
            y = list(p.map(f, x))

        y = tf.convert_to_tensor(y)
        return y


class BatchMultiChannelsJpegDNA(BatchSingleChannelJpegDNA):

    def reshape_input_encode(self, x):
        """Reshape the input to have the right shape for the encoder."""
        return tf.reshape(x, (-1, x.shape[1], x.shape[2] * x.shape[3], x.shape[4]))

    @encode_wrapper
    def encode_batch(self, x):
        """Encode a batch of images with several channels into a list of oligos."""
        # Input shape: (batch, height, width, channels)
        # Output shape: (batch, channels, nb_oligos)
        assert (
                len(x.shape) == 4
                ), "x must be a 4D tensor (batch, height, width, channels)"

        global _encode_worker

        def _encode_worker(x, alpha):
            """Encode a 2 dimensional array into several oligos."""
            return JpegDNA(alpha).encode(x.numpy(), "from_img")

        n_batches, n_channels = x.shape[0], x.shape[3]
        indexes = list(product(range(n_batches), range(n_channels)))
        f = partial(_encode_worker, alpha=self._alpha)
        with Pool(self.num_workers) as p:
            y = list(p.map(f, [x[i, :, :, j] for i, j in indexes]))

        # Reshape the tensor to (batch, channels, nb_oligos)
        y = [
                [y[i * n_channels + j] for j in range(n_channels)] for i in range(n_batches)
                ]
        return tf.ragged.constant(y, dtype=tf.string)

    @decode_wrapper
    def decode_batch(self, x, shape):
        """Decode a batch of oligos into a batch of images with several channels."""
        # Input shape: (batch, channels, nb_oligos)
        # Output shape: (batch, height, width, channels)
        assert len(x.shape) == 3, "x must be a 3D tensor (batch, channels, nb_oligos)"

        global _decode_worker

        def _decode_worker(oligos, alpha):
            """Decode a list of oligos into a 2 dimensional array."""
            return JpegDNA(alpha).decode(oligos.numpy().astype(str))

        n_batch, n_channels, _ = (
                x.bounding_shape() if type(x) == tf.RaggedTensor else x.shape
                )
        indexes = product(range(n_batch), range(n_channels))
        f = partial(_decode_worker, alpha=self._alpha)
        with Pool(self.num_workers) as p:
            y = list(p.map(f, [x[i, j] for i, j in indexes]))
        y = tf.convert_to_tensor(y)
        # Reshape the tensor to (batch, channels, height, width)
        y = tf.stack(tf.split(y, n_batch.numpy()))
        # Swap axes to have (batch, height, width, channels)
        return tf.transpose(y, [0, 2, 3, 1])


class CompressionModel(tf.keras.Model):
    """Main model class."""

    def __init__(self, args, name="CompressionModel"):
        super().__init__(name=name)
        self._args = args

        self.prior = tfc.NoisyDeepFactorized(batch_shape=[args.latent_depth])

        self.analysis_transform = AnalysisTransform(args.num_filters, args.latent_depth)
        self.synthesis_transform = SynthesisTransform(args.num_filters)
        self.codec, self.pack_tensor, self.unpack_tensor = parse_codec(args.codec, args.alpha)

        if args.transfer_learning_model is not None:

            # Initialize the weights of the analysis transform
            a = self.analysis_transform.call(tf.random.normal([1, 64, 64, 64, 1]))
            s = self.synthesis_transform.call(a)

            logger.info("Loading transfer learning model...")
            with tf.device("/cpu:0"):
                transfer_model = tf.keras.models.load_model(
                        args.transfer_learning_model
                        )

                self.analysis_transform.set_weights(
                        transfer_model.analysis_transform.weights
                        )

                # Have to set manually the weights of the synthesis transform
                # because the weights are not stored in the same order
                self.synthesis_transform.block1.set_weights(
                        transfer_model.synthesis_transform.block1.weights
                        )
                self.synthesis_transform.block2.set_weights(
                        transfer_model.synthesis_transform.block2.weights
                        )
                self.synthesis_transform.conv1.set_weights(
                        transfer_model.synthesis_transform.conv1.weights
                        )
                self.synthesis_transform.conv2.set_weights(
                        transfer_model.synthesis_transform.conv2.weights
                        )
                self.synthesis_transform.conv3.set_weights(
                        transfer_model.synthesis_transform.conv3.weights
                        )
                logger.info("Transfer learning model loaded.")

    def compile(self, optimizer, loss):
        super().compile(optimizer=optimizer, loss=loss)
        self.focal_loss = tf.metrics.Mean(name="focal_loss")
        self.nucleotides_rate = tf.metrics.Mean(name="nucleotides_rate")

    def dna_encoding(self, x, apply_dct=False):
        """Encodes the latent blocks to DNA oligos."""
        assert (
                len(x.shape) == 5
                ), "The input must be of shape [batch_size, b1, b2, b3, latent_depth]."
        shape = batch_size, b1, b2, b3, latent_depth = tf.shape(
                x
                )  # TODO change: the way to store the mid shape

        # Quantize the blocks
        quantized_x, quantize_ranges = quantize(x)

        # Encode the images to DNA oligos
        # logger.info("Encoding the latent blocks to DNA oligos...")
        with tf.device("/cpu:0"):
            oligos = self.codec.encode_batch(tf.cast(quantized_x, tf.int32), apply_dct=apply_dct)

        return oligos, shape[1:], quantize_ranges

    def dna_decoding(self, oligos, shape, quantize_ranges, apply_dct=False):
        """Decodes the DNA oligos to latent blocks."""

        # Decode the DNA oligos to images
        # logger.info("Decoding the DNA oligos to latent blocks...")
        with tf.device("/cpu:0"):
            quantized_x = self.codec.decode_batch(oligos, shape, apply_dct=apply_dct)


        # Dequantize the blocks
        quantized_x = tf.cast(quantized_x, tf.float32)
        x = dequantize(quantized_x, quantize_ranges)

        return x

    def compress(self, x, full_output=True, apply_dct=False):

        # Build the encoder (analysis) half of the hierarchical autoencoder.
        y = self.analysis_transform(x)

        # Build the bottleneck
        z, shape, quantize_ranges = self.dna_encoding(y, apply_dct=apply_dct)

        if full_output:
            geo_x = x[:, :, :, :, 0]

            num_voxels = reduce_batch(tf.reduce_sum, tf.ones(geo_x.shape))
            num_occupied_voxels = reduce_batch(tf.reduce_sum, geo_x)

            info = Namespace({
                "y": y,
                "num_voxels": num_voxels,
                "num_occupied_voxels": num_occupied_voxels,
                })
            return z, shape, quantize_ranges, info
        return z, shape, quantize_ranges

    def decompress(self, z, y_shape, quantize_ranges, full_output=True, apply_dct=False):
        batch_size, *_ = z.shape
        # Build the decoder (synthesis) half of the hierarchical autoencoder.
        y_hat = self.dna_decoding(z, y_shape, quantize_ranges, apply_dct=apply_dct)

        # Need to add the batch dimension if is equal to 1 because tf.quantize squeezes it
        # if batch_size == 1:
        #     y_hat = tf.expand_dims(y_hat, 0)

        # Build the bottleneck
        x_hat = self.synthesis_transform(y_hat)

        if full_output:
            return x_hat, Namespace({'y_hat': y_hat})
        return x_hat

    def call(self, x, apply_dct=False):
        """Computes distortion loss."""

        # Compute the bottleneck
        z, shape, quantize_ranges, compression_info = self.compress(x, apply_dct=apply_dct)

        x_hat, decompression_info = self.decompress(z, shape, quantize_ranges, apply_dct=apply_dct)

        # Compute the focal loss and/or color loss across pixels.
        # Don't clip or round pixel values while training.
        fcl = focal_loss(x, x_hat, gamma=2, alpha=self._args.alpha) / num_voxels
        nucleotide_rate = number_of_nucleotides(z) / num_voxels
        loss = nucleotide_rate + self._args.lmbda * fcl
        info = {
                **compression_info,
                **decompression_info,
                "focal_loss": fcl,
                "nucleotide_rate": nucleotide_rate,
                }
        return loss, info

    def train_step(self, x):
        raise NotImplementedError

    def test_step(self, x):
        """Performs a test step."""
        loss, info = self.call(x)
        self.focal_loss.update_state(loss)
        return {m.name: m.result() for m in [self.focal_loss]}


def load_model(args):
    """Loads the model."""
    if args.model_checkpoint != "":
        return tf.keras.models.load_model(args.model_checkpoint)
    return CompressionModel(args)

def compress(model, args):
    global ds, z, y_shape, quantize_ranges, info_compress 
    """Compress the dataset"""
    def validate_args(args=args):
        if np.isreal(args.threshold) and args.save_intermediate:
            logger.warning(
                    "The `save_intermediate` flag is set to True, but the ",
                    "`threshold` is not set to `adaptive` so the intermediate ",
                    "results will not be saved."
                    )

        if not args.apply_dct and args.gammas is not None:
            logger.warning(
                    "The `apply_dct` flag is set to False, but the ",
                    "`gammas` flag is not None so the gammas will not be ",
                    "applied."
                    )

    validate_args(args)

    # Set the number of workers to use for the codec
    model.codec.num_workers = args.num_workers

    # Set the gammas values used in the dct quantization if applicable
    model.codec.gammas = args.gammas

    # Create a tensorflow dataset from the point clouds.
    ds = pc_dir_to_ds(
            args.io.input,
            args.blocks.resolution,
            args.blocks.channels_last,
            ).batch(args.num_workers)


    for data in tqdm(ds, total=ds.cardinality().numpy()):
        x = data["input"]
        names = [n.numpy().decode("UTF-8").split("/")[-1].split(".")[0] for n in data["fname"]]
        out_fnames = [os.path.join(args.io.output, name + ".dna") for name in names]
        batch_size = x.shape[0]

        # Skip if the output file already exists
        # if os.path.exists(out_fname) and not args.io.overwrite:
        #     logger.info(f"Because compress.io.overwrite: skipping {out_fname}")

        z, y_shape, quantize_ranges, info_compress = model.compress(x, apply_dct=args.apply_dct)

        # Calculate the adaptive threshold.
        if args.threshold == "adaptive":
            x_hat, info_decompress = model.decompress(z, y_shape, quantize_ranges, apply_dct=args.apply_dct)
            logger.info("Calculating the adaptive threshold...")
            with Pool(args.num_workers) as pool:
                f = partial(compute_optimal_threshold, 
                    delta_t=0.01,
                    breakpt=150,
                    verbose=1,
                    )
                ret = list(pool.starmap(f, zip(x_hat, data["pc"].numpy())))
            thresholds, pa = zip(*ret)


            if args.io.save_intermediate_results != "":
                # Save the intermediate results in another thread.
                save_all_intermediate_results(names, info_compress.y, info_decompress.y_hat, pa, args.io.save_intermediate_results, join_thread=False)

        else:
            assert 0.0 < args.threshold < 1.0, "Threshold must be between 0 and 1."
            thresholds = tf.constant(args.threshold, shape=(batch_size,))

        logger.info("Pack the representations...")
        nucleotide_streams = [model.pack_tensor(
                threshold, model.codec.oligo_length, quantize_range, y_shape, z_strings
                ) for threshold, quantize_range, z_strings in zip(thresholds, quantize_ranges, z)]

        logger.info("Saving the compressed data to %s", args.io.output)

        # Save the compressed representations in another thread.
        save_oligos(out_fnames, nucleotide_streams, args.io.output, join_thread=False)
        # os.makedirs(args.io.output, exist_ok=True)
        # for stream, out_fname in zip(nucleotide_streams, out_fnames):
        #     with open(out_fname, "w+") as f:
        #         f.write(stream)


def decompress(model, args):
    """Decompress the dataset"""
    global ds

    files = pc_io.get_files(args.io.input)

    for fname in tqdm(files):
        name = fname.split("/")[-1].split(".")[0]

        out_fname = os.path.join(args.io.output, name + ".dna")
        if os.path.exists(out_fname) and not args.io.overwrite:
            logger.info(f"Because decompress.io.overwrite: skipping {out_fname}")
            continue

        with open(fname, "r") as fd:
            nucleotide_stream = fd.read()

        threshold, _, y_shape, z_strings = model.unpack_tensor(
                nucleotide_stream,
                )

        # Reconstruct the point clouds.
        logger.info("Reconstructing the point clouds...")
        x_hat = model.decompress(tf.expand_dims(z_strings, 0), y_shape).numpy()[0]

        pa = np.argwhere(x_hat[..., 0] > threshold.numpy()).astype(np.float32)

        # Save the reconstructed point clouds.
        os.makedirs(args.io.output, exist_ok=True)

        pc_io.write_df(args.io.output.rstrip("/*") + f"/{name}.ply", pc_io.pa_to_df(pa))


@hydra.main(config_path="config/main", config_name="default.yaml", version_base="1.2")
def main(cfg):
    global args, model
    args = omegaconf2namespace(cfg, allow_missing=False)

    # If gpu needed, set the memory growth option so it does not allocate all the memory.
    if args.gpu:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    # Otherwise, set the CPU as the default device.
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    tasks = ["compress", "decompress"]
    if args.task not in ["compress", "decompress", "play"]:
        raise ValueError(f"Unknown task: {args.task}, choose between {tasks}")

    # Load the model
    model = load_model(args.architecture)

    if args.task == "compress":
        compress(model, args.compress)
    elif args.task == "decompress":
        decompress(model, args.decompress)



if __name__ == "__main__":
    main()
