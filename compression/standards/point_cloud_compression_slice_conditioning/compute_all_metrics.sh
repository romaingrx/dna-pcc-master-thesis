#!/bin/bash

IFS=$'\n'
for exp in `ls results | grep -e 'a0.6_res_t64_Slice_cond_160-10_.*'`; do
    python compute_metrics.py experiment=$exp
done

