<p float="left"> <img src="./img/logos/logos.png"> </p>

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/) [![CircleCI](https://circleci.com/gh/jpegdna-mediacoding/Jpeg_DNA_Python/tree/master.svg?style=shield&circle-token=3dfba08233b9ae9bad08091b12bac0bccafd060d)](https://circleci.com/gh/jpegdna-mediacoding/Jpeg_DNA_Python/tree/master) [![codecov](https://codecov.io/gh/jpegdna-mediacoding/Jpeg_DNA_Python/branch/master/graph/badge.svg?token=M87ZLKHJ9Q)](https://codecov.io/gh/jpegdna-mediacoding/Jpeg_DNA_Python)

# Description

This repository aims to be an example of software for DNA image coding in agreement with the standards described by the Jpeg DNA research group.

This library contains codecs and scripts for compressing and encoding images into quaternary DNA like data streams. These streams are not written in binary code but in quaternary code, using the DNA symbols (A, T, C, G) called nucleotides as bases.

This project has been developped by the Mediacoding group in the I3S laboratory (Euclide B, 2000 Rte des Lucioles, 06900 Sophia Antipolis, FRANCE), with fundings from CNRS and Université Côte d'Azur.

# Contact the authors

I3S laboratory, Euclide B, 2000 Rte des Lucioles, 06900 Sophia Antipolis, FRANCE

xpic@i3s.unice.fr, dimopoulou@i3s.unice.fr, gilsanan@i3s.unice.fr, am@i3s.unice.fr

# Installation

To install the repository, you need to have `pip` installed, then use the following command at the root of the folder:

> `pip install -r requirements.txt`

or if you want to add the jpegdna package to your python path and use it from anywhere:

> `python setup.py install`

# Jpeg DNA commands usage

In order to use the commands from the terminal, you first need to install the package:

> `python setup.py install`

## Encoding

If you are using the image data to compute the frequencies of the DCT coefficients, you can run the encoding with:

> `jdnae $IMG_FILE_PATH $DNA_OUT_FILE_PATH.pkl $alpha`

or

> `jdnae.exe $IMG_FILE_PATH $DNA_OUT_FILE_PATH.pkl $alpha`

otherwise:

> `jdnae -d $IMG_FILE_PATH $DNA_OUT_FILE_PATH.pkl $alpha`

or

> `jdnae.exe -d $IMG_FILE_PATH $DNA_OUT_FILE_PATH.pkl $alpha`

## Decoding

Use:

> `jdnad $DNA_IN_FILE_PATH.pkl $IMG_OUT_FILE_PATH`

or

> `jdnad.exe $DNA_IN_FILE_PATH.pkl$IMG_OUT_FILE_PATH`

# Jpeg DNA scripts usage

This implementation for the moment only deals with encoding and formatting the data payload for a gray-level image. It does not support the formatting of the variables necessary for decoding (frequency values of the DCT and size of the image).

Some scripts have been implemented to use the Jpeg DNA codec, more specifically, one for encoding and one for decoding. The scripts can be executed in the following way:

## Gray level

### For encoding

If you are using the image data to compute the frequencies of the DCT coefficients, you can run the encoding with:

> `python -m jpegdna.scripts.jpegdna_encode $IMG_PATH $DNA_OUT_PATH $ALPHA`

If you want to use the default frequencies for the DCT coefficient, you can add the '-d' option:

> `python -m jpegdna.scripts.jpegdna_encode $IMG_PATH $DNA_OUT_PATH $ALPHA -d`

If you want to enable the formatting system, you can add the '-f' option (also works in combination with the '-d' option):

> `python -m jpegdna.scripts.jpegdna_encode $IMG_PATH $DNA_OUT_PATH $ALPHA -f` `python -m jpegdna.scripts.jpegdna_encode $IMG_PATH $DNA_OUT_PATH $ALPHA -d -f`

### For decoding

If you are using non formatted data, to decode the image, you will still need to specify the alpha value (the frequencies and the image dimensions are transmitted through files that are specified in jpegdna/scripts/config.ini):

> `python -m jpegdna.scripts.jpegdna_decode $DNA_IN_PATH $IMG_OUT_PATH no_format $ALPHA`

If you want to use the default frequencies for the DCT coefficient, you can add the '-d' option before the no_format option:

> `python -m jpegdna.scripts.jpegdna_decode $DNA_IN_PATH $IMG_OUT_PATH -d no_format $ALPHA`

Otherwise, if the data is formatted, there is no need to specify the alpha value or if the frequencies are the preconfigured one because all the information necessary for decoding is already available in the format:

> `python -m jpegdna.scripts.jpegdna_decode $DNA_IN_PATH $IMG_OUT_PATH`

### Evaluation script

An evaluation script is available under scripts/jpegdna_eval.py . The image to be tested can be specified at the beginning of the __name__ handler. The script will run encoding and decoding for different dynamic (alpha) values and get the compression rate and PSNR for the whole process. The dynamic values can be modified by the user in jpegdna_eval.py. An exception is for the moment used to check if the dynamic is not to small.

It can be executed from the root folder with:

> `python -m jpegdna.scripts.jpegdna_eval`

## RGB

### For encoding

If you are using the image data to compute the frequencies of the DCT coefficients, you can run the encoding with:

> `python -m jpegdna.scripts.jpegdnargb_encode $IMG_PATH $DNA_OUT_PATH $ALPHA`

If you want to use the default frequencies for the DCT coefficient, you can add the '-d' option:

> `python -m jpegdna.scripts.jpegdnargb_encode $IMG_PATH $DNA_OUT_PATH $ALPHA -d`

If you want to enable the formatting system, you can add the '-f' option (also works in combination with the '-d' option):

> `python -m jpegdna.scripts.jpegdnargb_encode $IMG_PATH $DNA_OUT_PATH $ALPHA -f` `python -m jpegdna.scripts.jpegdnargb_encode $IMG_PATH $DNA_OUT_PATH $ALPHA -d -f`

### For decoding

If you are using non formatted data, to decode the image, you will still need to specify the alpha value (the frequencies and the image dimensions are transmitted through files that are specified in jpegdna/scripts/config.ini):

> `python -m jpegdna.scripts.jpegdnargb_decode $DNA_IN_PATH $IMG_OUT_PATH no_format $ALPHA`

If you want to use the default frequencies for the DCT coefficient, you can add the '-d' option before the no_format option:

> `python -m jpegdna.scripts.jpegdnargb_decode $DNA_IN_PATH $IMG_OUT_PATH -d no_format $ALPHA`

Otherwise, if the data is formatted, there is no need to specify the alpha value or if the frequencies are the preconfigured one because all the information necessary for decoding is already available in the format:

> `python -m jpegdna.scripts.jpegdnargb_decode $DNA_IN_PATH $IMG_OUT_PATH`

### Evaluation script

An evaluation script is available under scripts/jpegdna_eval.py . The image to be tested can be specified at the beginning of the __name__ handler. The script will run encoding and decoding for different dynamic (alpha) values and get the compression rate and PSNR for the whole process. The dynamic values can be modified by the user in jpegdna_eval.py. An exception is for the moment used to check if the dynamic is not to small.

It can be executed from the root folder with:

> `python -m jpegdna.scripts.jpegdnargb_eval`

## Verbosity

A verbosity parameter can be adjusted for both encoding and decoding in the jpegdna/scripts/config.ini file under the VERB section for encoding and decoding. If `enabled = False`, the script wont print anything, if it is set to True, the `level` parameter will adjust the amount of information accessible:

| Level | Effects | |-------|-----------------------------------------------------------------------------------------------------------| | 0 | Input image, output strand printed during encoding, input strand, output image printed during decoding | | 1 | Basic info for encoding and decoding at the block level | | 2 | Basic info for quantization, zigzag, and DCT results on blocks | | 3 | Basic info for encoding and decoding at the value level | | 4 | Additional info for the HuffmanCoder class, helpful for debugging | | 5 | Additional info for the ValueCoder, helpful for debugging |

The evaluation script is configurable directly in the script, where the codec is instanciated `JPEGDNA(alpha, verbose=True, verbosity=0)` or `JPEGDNARGB(alpha, verbose=True, verbosity=0)`.

# Features and Jpeg standard components not yet available

* Headers containing the quantization tables used for compression are not yet included
* Headers containing the Goldman codewords of the categories and run/categories are not yet available.
* Jpeg and JpegDNA are not interchangeble when transcoding the compressed quaternary stream into binary.
* If there is an error in the deformatted stream (cat/runcat/value sequence) the code will stop. This is due to the fact that the code we use to encode the values into DNA is not complete. It means that when the image quaternary stream is noised, if the noised codeword is not a valid codeword (not present into the codebook), the codeword is not decodable and the algorithm stops.
* No consensus has been uploaded yet to deformat noised oligos.
* No error correction has been uploaded yet.