#!/bin/bash

source ~/.thesis_config.sh

block_size=128
keep_size=1
experiment="a0.6_res_t64_Slice_cond_160-10_40"
model_dir="$(pwd)/models/"

data_dir="$(pwd)/data"
data_names=(
    "${data_dir}/original"
    "${data_dir}/original_blocks"
    "${data_dir}/compressed_blocks"
    "${data_dir}/reconstruct_blocks"
    "${data_dir}/reconstruct"
    )

partiton_exe="${thesis}/compression/standards/point_cloud_compression_geometry_color/pcc-geo-color/src/partition.py"
merge_exe="${thesis}/compression/standards/point_cloud_compression_geometry_color/pcc-geo-color/src/merge.py"
pcc_slice_cond_exe="${thesis}/compression/standards/point_cloud_compression_slice_conditioning/pcc-geo-slicing/point_cloud_compression_slice_conditioning.py"

# Creation of all the directories
for dir in "${data_names[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
    fi
done


# Transform the raw data into blocks
python "${partiton_exe}" \
    "${data_names[0]}" \
    "${data_names[1]}" \
    --block_size "$block_size" \
    --keep_size "$keep_size"

# Compress the blocks
python ${pcc_slice_cond_exe} \
    --model_path ${model_dir} \
    --experiment ${experiment} \
    compress \
    --input_glob "${data_names[1]}/*" \
    --output_dir "${data_names[2]}/" \
    --resolution ${block_size}

# Uncompress the blocks
python ${pcc_slice_cond_exe} \
    --model_path ${model_dir} \
    --experiment ${experiment} \
    decompress \
    --ori_dir "${data_name[1]}" \
    --input_glob "${data_names[2]}/*" \
    --output_dir "${data_names[3]}/" \
    --resolution ${block_size}

# Merge the reconstructed blocks 
python ${merge_exe} \
    "${data_names[0]}/" \
    "${data_names[3]}/" \
    "${data_names[4]}/" \
    --resolution "$block_size" \
    --task geometry
