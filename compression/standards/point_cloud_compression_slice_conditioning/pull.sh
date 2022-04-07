#!/bin/bash


usage(){
    echo "Usage: $0 {data,models}"
    exit 1
}

if [ $# -ne 1 ]; then
    usage
fi

if [ $1 == "data" ]; then
    echo "Downloading data..."
    wget https://nextcloud.romaingraux.xyz/s/qNTPzMiXfYHD2eQ/download -O data.zip
    unzip data.zip
    rm data.zip
    echo "Done."
elif [ $1 == "models" ]; then
    echo "Downloading models..."
    wget https://nextcloud.romaingraux.xyz/s/HqQoKFEzP4jRWw2/download -O models.zip
    unzip models.zip
    rm models.zip
    echo "Done."
else
    usage
fi
