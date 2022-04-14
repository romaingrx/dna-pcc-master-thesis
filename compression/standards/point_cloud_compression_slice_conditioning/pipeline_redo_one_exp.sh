#!/bin/bash

source ./pipeline_compress_reconstruct.sh

if [ $# -lt 1 ]; then
    exit 1
fi

data_dir="$1"


experiment=`echo $data_dir | awk '{printf $3}' | awk -F '/' '{print $1}'`
point_cloud_name=`echo $data_dir | awk '{printf $3}' | awk -F '/' '{print $2}'`
original_point_cloud_path="./data/original/${point_cloud_name}"
log_file="${data_dir}/log.txt"

compress_reconstruct_pipeline "$experiment" "$original_point_cloud_path" "$data_dir" "$log_file"
