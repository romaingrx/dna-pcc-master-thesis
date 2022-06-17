<div>
    <img height="75" src="docs/imgs/epfl-logo.svg">
    <img align="right" height="75" src="docs/imgs/mmspg-logo.png">
</div>

<h1 align="center">Point cloud compression for DNA based storage</h1>

> This repository contains all the code developped for my master thesis in the [MMPSG](https://www.epfl.ch/labs/mmspg) lab supervised by Dr. Prof. [Touradj Ebrahimi](http://people.epfl.ch/touradj.ebrahimi) and [Davi Nachtigall Lazzarotto](https://people.epfl.ch/davi.nachtigalllazzarotto). This project has been done during the spring semester 2022 at [EPFL](https://www.epfl.ch). 

The purpose of this project is to build an end-to-end point cloud compressor for quaternary based coding while respecting DNA sequencing, storage and synthesis constraints.
This project is based on several publicly available implementations for the [point cloud compression](https://github.com/mmspg/pcc-geo-slicing), the [DNA entropy coding](codecs/jpeg_dna_codec) and a [flexible DNA error detection and simulation framework](https://github.com/umr-ds/mesa_dna_sim).


DNA point cloud compression implementation
---

> The DNA point cloud compression code can be found [here](compression/dna/codec), go there to use the scripts.

<details>
    <summary>Installation</summary>   
In order to be able to play with the compressor, you will need to first install some dependencies.

First create a new virtual environment with `python=3.8.10`

Then, install all the requirements with the command:

``` shell
pip install -r requirements.txt
``` 
</details>
<details>
    <summary>Download the backbone model, dataset or results</summary>
    
For this project you will need at least one of the model trained for the classical point cloud compression as a backbone, you can download the model with the best quality with the command:
``` bash
bash pull.sh model
```

If you need already voxelized dataset, you can download it with the command:
``` bash
bash pull.sh dataset
```
    
And if you desire to download the results presented in the [report](report/main.pdf), you can execute:
``` bash
bash pull.sh results
```
    
</details>

This project is using [hydra](https://hydra.cc) as a configuration handler, all the config files can be found under the [config](compression/dna/codec/config) folder, in this folder you can find a subfolder with the default configuration for each task.
When you do not know how a particular python files can be configured, you can simply pass the `--help` flag while executing the script, [hydra](https://hydra.cc) will output all the configurable parameters.

You can always override a parameter by adding the parameters as a flag with its value, a full documentation can found [here](https://hydra.cc/docs/configure_hydra/intro).

The folder contains several python files for different purposes.

The [main.py](compression/dna/codec/main.py) contains the main code to `compress` and `decompress` a point cloud dataset into DNA stream.

### Compress

Before compressing a full point cloud, you first have to voxelize it and then partition it with the script [partition.py](bin/partition.py) into smaller resolution (usually 64 or 128) point cloud blocks.

<details>
    <summary>How to?</summary>
    
``` bash 
python partition.py \
    --block_size 64 \ # The resolution of each block 
    --keep_size 500 \ # The minimum number of voxels in a block under which we drop it
    "${datasets_dir}/vox8/ply" \ # The input directory with the full voxeliezd point clouds
    "${datasets_dir}/vox8/blocks_64" # The output directory that will contain the blocks
```
</details>

Then to compress a folder containg several blocks in `.ply` format into a folder with the corresponding `.dna` streams for each point cloud block, you can for example execute:

``` bash
python main.py \
    task="compress" \
    experiment="a0.6_res_t64_Slice_cond_160-10_1750" \ # One of the model to use as backbone for compression
    ++results_dir="$(pwd)/results" \
    ++compress.io.input="${datasets_dir}/vox8/ply" \ # The folder should contains all *.ply files
    ++compress.num_workers=20 \ # The number of cores to parallelize the compression
    ++compress.blocks.resolution=64 \ # The resolution of the blocks in the input folder
    ++compress.quantization_span=1000 # The maximum quantized value to control the nucleotide rate (between 0 and 17579)
```

> All the lines starting with a `++` are optionals

### Decompress

Then to decompress a folder containg several DNA streams in `.dna` format into a folder with the corresponding `.ply` reconstructed the corresponding point cloud blocks, you can for example execute:

``` bash
python main.py \
    task="decompress" \
    experiment="a0.6_res_t64_Slice_cond_160-10_1750" \ # One of the model to use as backbone for decompression
    ++results_dir="$(pwd)/results" \
    ++decompress.blocks.resolution=64 \ # The resolution of the blocks in the input folder
    ++decompress.quantization_span=1000 # The maximum quantized value to control the nucleotide rate (between 0 and 17579)
```

> All the lines starting with a `++` are optionals

Once all blocks have been reconstructed you can reassemble them to the final point clouds with the script [merge.py](bin/merge.py)

<details>
    <summary>How to?</summary>
    
``` bash 
python merge.py \
    python $merge_exe --resolution 64 --task geometry
    --resolution 64 \ # The resolution of each block 
    --task geometry \ # Can be in {geometry,color,geometry+color} but in our case, only geometry is needed
    "${datasets_dir}/vox8/ply" \ # The directory with the original voxelized point clouds
    "$(pwd)/results/x_hat" # The directory containing the blocks 
    "$(pwd)/results/reconstructed_pc" # The directory that will contain the full merged point clouds
```
</details>

### Simulate

To simulate DNA sequences, a small class has been developped in the file [simulator.py](compression/dna/codec/simulator.py) to interact with the MESA simulator (local or online).
This class allows to send sequences or fasta files, fetch the already simulated sequences with their UUID or from a fastq file that the server sent to your emails.

The config of the simulation can be modified [here](compression/dna/codec/config/simulator/post/default.yaml).

To retrieve the modified sequences after having received the fastq by email, you can execute:

``` bash 
python simulator.py \
    ++connection.host="mesa.mosla.de" \ # The host of the MESA server ("localhost" if you run a local docker)
    ++connection.port="" \ # For example "", 80 (http) or 443 (https)
    ++connection.secure=true \ # true if https, false if http
    ++post.key="${YOUR_API_KEY}" \ # The key that the server gave you in the 
    ++connection.n_workers=20 \ # The number of threads fetch data from the server in parallel
    ++fasta.root="$(pwd)/fasta" # The root directory with the 'in' (fasta files), 'simulated' (the fastq files received by the server) and 'out' (the modified sequences fetched from the server) folders
``` 

> All the lines starting with a `++` are optionals
