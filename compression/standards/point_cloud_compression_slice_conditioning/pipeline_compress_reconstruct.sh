#!/bin/bash

source ~/.thesis_config.sh

usage(){    
    echo $0 experiment point_cloud_path
}

compress_reconstruct_pipeline(){
    experiment="$1"
    point_cloud_path="$2"
    data_dir="$3"
    log_file="$4"

    block_size=128
    keep_size=1
    model_dir="$(pwd)/models/"
    
    data_names=(
        ${point_cloud_path}
        "${data_dir}/original_blocks"
        "${data_dir}/compressed_blocks"
        "${data_dir}/reconstruct_blocks"
        "${data_dir}/reconstruct"
        )
    
    partiton_exe="${thesis_dir}/compression/standards/point_cloud_compression_geometry_color/pcc-geo-color/src/partition.py"
    merge_exe="${thesis_dir}/compression/standards/point_cloud_compression_geometry_color/pcc-geo-color/src/merge.py"
    pcc_slice_cond_exe="${thesis_dir}/compression/standards/point_cloud_compression_slice_conditioning/pcc-geo-slicing/point_cloud_compression_slice_conditioning.py"
    
    # Creation of all the directories
    for dir in "${data_names[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
        fi
    done
    
    echo -ne "    ... 1/4 Partitioning the point cloud\r                      "
    
    # Transform the raw data into blocks
    python "${partiton_exe}" \
        "${data_names[0]}" \
        "${data_names[1]}" \
        --block_size "$block_size" \
        --keep_size "$keep_size" \
        &> "${log_file}"

    echo -ne "    ... 2/4 Compressing the point cloud\r                      "
    
    # Compress the blocks
    python ${pcc_slice_cond_exe} \
        --model_path ${model_dir} \
        --experiment ${experiment} \
        compress \
        --adaptive \
        --input_glob "${data_names[1]}/*" \
        --output_dir "${data_names[2]}/" \
        --resolution ${block_size} \
        &>> "${log_file}"

    echo -ne "    ... 3/4 Reconstructing the point cloud\r                      "
    
    # Uncompress the blocks
    python ${pcc_slice_cond_exe} \
        --model_path ${model_dir} \
        --experiment ${experiment} \
        decompress \
        --adaptive \
        --ori_dir "${data_name[1]}" \
        --input_glob "${data_names[2]}/*" \
        --output_dir "${data_names[3]}/" \
        --resolution ${block_size} \
        &>> "${log_file}"

    echo -ne "    ... 4/4 Merging the point cloud ${data_names[4]}"

    # Merge the reconstructed blocks 
    python ${merge_exe} \
        "${data_names[0]}/" \
        "${data_names[3]}/" \
        "${data_names[4]}/" \
        --resolution "$block_size" \
        --task geometry
}

if [ $# -eq 2 ]; then
    experiments_dir=$1
    original_point_cloud_path=$2
    
    n_exp=1
    for experiment in $(ls ${experiments_dir}); do
        echo -e "\e[1;34mExperiment [$n_exp/$(ls $experiments_dir | wc -l)]:\e[0m ${experiment}"
        uuid="$(date +"%Y-%m-%d %H:%M:%S") ${experiment}"
        n_pc=1
        for point_cloud_name in $(ls $original_point_cloud_path); do
            data_dir="$(pwd)/data/${uuid}/${point_cloud_name}"
            mkdir -p "$data_dir"
            log_file="${data_dir}/log.txt"
            echo -e "    > \e[1;35mPoint cloud [$n_pc/$(ls $original_point_cloud_path | wc -l)]:\e[0m $point_cloud_name"
            compress_reconstruct_pipeline "$experiment" "$original_point_cloud_path/$point_cloud_name" "$data_dir" "$log_file"
            n_pc=$((n_pc+1))
        done 
        n_exp=$((n_exp+1))
    done 
fi
