#!/bin/bash


CMD="$1"
RESULT_DIR="${2:-./results}"
GREP_PATTERN="${3:-a0.6_res_t64_Slice_cond_160-10_.*}"

IFS=$'\n'
cwd=`pwd`
for exp in `ls ${RESULT_DIR} | grep -e "$GREP_PATTERN"`; do
    cd ${cwd}/${RESULT_DIR}/${exp}
    eval ${CMD}
done
