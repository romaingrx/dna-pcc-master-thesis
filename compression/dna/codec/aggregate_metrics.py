#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 May 31, 18:07:48
@last modified : 2022 May 31, 18:41:50
"""

import os
import hydra
import pandas as pd
from helpers import omegaconf2namespace

def gather_df_metrics(fname, aggregations):
    """
    Gather metrics from a dataframe
    """
    df = pd.read_csv(fname)
    df['point_cloud_name'] = df['name'].apply(lambda x: x.split('_')[0])
    aggregated = df.groupby('point_cloud_name').agg(aggregations)
    return aggregated

@hydra.main(config_path="config/aggregate_metrics", config_name="default.yaml")
def main(cfg):
    global aggregated, args, fname, global_df
    args = omegaconf2namespace(cfg)
    global_df = pd.DataFrame()
    for experience in os.listdir(args.io.experiences_dir):
        fname = os.path.join(args.io.experiences_dir, experience, 'metrics.csv')
        if os.path.exists(fname):
            aggregated = gather_df_metrics(fname, args.aggregation_functions)
            aggregated['experience'] = experience
            global_df = global_df.append(aggregated)
    global_df.to_csv(args.io.output_file, index=False)

if __name__ == '__main__':
    main()
