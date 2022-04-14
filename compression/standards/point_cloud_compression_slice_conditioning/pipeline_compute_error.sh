#!/bin/bash



source ~/.thesis_config.sh
tmpfile=$(mktemp)

block_size=128
error_exe="$thesis_dir/compression/mpeg-pcc-dmetric/test/pc_error"

# Output the csv line "experiment,point_cloud,mse1,mse1PSNR,mse2,mse2PSNR,mse,msePSNR"
compute_error_to_csv_line() {
    fileA=$1
    fileB=$2
    resolution=$3
    nbThreads=$4

    $error_exe --fileA="$fileA" --fileB="$fileB" --resolution=$resolution --nbThreads=$nbThreads > $tmpfile
    mse1=`cat $tmpfile | grep -o -E 'mse1\s.*' | grep -o -E '[0-9]+\.[0-9]+'`
    mse1PSNR=`cat $tmpfile | grep -o -E 'mse1\,PSNR.*' | grep -o -E '[0-9]+\.[0-9]+'`
    mse2=`cat $tmpfile | grep -o -E 'mse2\s.*' | grep -o -E '[0-9]+\.[0-9]+'`
    mse2PSNR=`cat $tmpfile | grep -o -E 'mse2\,PSNR.*' | grep -o -E '[0-9]+\.[0-9]+'`
    mseF=`cat $tmpfile | grep -o -E 'mseF\s.*' | grep -o -E '[0-9]+\.[0-9]+'`
    mseFPSNR=`cat $tmpfile | grep -o -E 'mseF\,PSNR.*' | grep -o -E '[0-9]+\.[0-9]+'`
    pc_sizes=`cat $tmpfile | grep "Point cloud sizes" | grep -E -o "[0-9]+\, [0-9]+\, [0-9]+\.[0-9]+"`
    org_pc_size=`echo $pc_sizes | awk -F ', ' '{print $1}'`
    dec_pc_size=`echo $pc_sizes | awk -F ', ' '{print $2}'`
    org_file_size=`du -b $fileA | awk '{print $1}'`
    dec_file_size=`du -b $fileB | awk '{print $1}'`

    echo "$mse1,$mse1PSNR,$mse2,$mse2PSNR,$mseF,$mseFPSNR,$org_pc_size,$dec_pc_size,$org_file_size,$dec_file_size"
}

echo "experiment,point_cloud,mse1,mse1PSNR,mse2,mse2PSNR,mse,msePSNR,org_pc_size,dec_pc_size,org_file_size,dec_file_size"

IFS=$'\n'
for experiment in `ls ./data | sort -r | grep '^2022'`; do
    exp_name=`echo $experiment | awk '{printf $3}'`
    for point_cloud in `ls ./data/original`; do
        original_file="./data/original/$point_cloud/$point_cloud.ply"
        reconstructed_file="./data/$experiment/$point_cloud/reconstruct/${point_cloud}_dec.ply"
        if [ -f "$reconstructed_file" ]; then
            line=`compute_error_to_csv_line $original_file $reconstructed_file $block_size 12`
            if [ $line == ",,,,,,,,,,," ]; then
                echo "Could not compute error for $reconstructed_file" >&2
            fi
            echo "$exp_name,$point_cloud,$line"
        else
            echo "Could not find $reconstructed_file" >&2
            echo "$exp_name,$point_cloud,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN"
        fi
    done
done
