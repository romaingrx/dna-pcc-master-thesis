#!/bin/bash

usage(){
    echo "Usage: $0 {model,dataset,results}"
    exit 1
}

if [ $# -ne 1 ]; then
    usage
fi

if [ $1 == "model" ]; then
    echo "Downloading backbone model to compression/dna/codec/models/pcc-geo-slicing..."
    mkdir -p compression/dna/codec/models/pcc-geo-slicing
    cd compression/dna/codec/models/pcc-geo-slicing
    wget https://cloud.romaingraux.xyz/s/RKJeXQooeHGrwMg/download -O model.zip
    unzip model.zip
    rm model.zip
    echo "Done."
elif [ $1 == "dataset" ]; then
    echo "Downloading the dataset to compression/datasets/..."
    cd compression/datasets
    wget https://cloud.romaingraux.xyz/s/DEn3qgiPPtYs3yw/download -O dataset.zip
    unzip dataset.zip
    rm dataset.zip
    echo "Done."
elif [ $1 == "results" ]; then
    echo "Downloading the results to compression/dna/codec/results (2GB)..."
    mkdir -p compression/dna/codec/results
    cd compression/dna/codec/results
    wget https://cloud.romaingraux.xyz/s/QYict5dqbWmfDgc/download -O results.zip
    unzip results.zip
    rm results.zip
    echo "Done."
else
    usage
fi
