#!/bin/bash


IFS=$'\n'

input_dir="./data"
output_dir="./results"

for raw_exp in `ls $input_dir | grep -e '.* a0.6_res_t64_Slice_cond_160-10_.*'`; do
    exp=`echo $raw_exp | awk -F "\ " '{print $3}'`
    # Create the dir in the output dir
    output_dir_exp=$output_dir/$exp
    mkdir -p $output_dir_exp
    for pc in `ls $input_dir/$raw_exp`; do
        dir="$input_dir/$raw_exp/$pc"
        if [ -d "$dir" ]; then
            for category in `ls $dir`; do
                output_dir_cat="$output_dir_exp/$category"
                mkdir -p $output_dir_cat
                if [ -d "$dir/$category" ]; then
                    cp -r $dir/$category/* $output_dir_cat/
                fi
            done
        fi
    done
done
