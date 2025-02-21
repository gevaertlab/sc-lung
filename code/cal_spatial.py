# Spatial statistical analysis

import anndata as ad
import pandas as pd
import numpy as np
import os
import scimap as sm
from natsort import natsorted
import matplotlib.pyplot as plt
from spatial_util import spatial_interaction_count
import warnings
import argparse
import pdb

source_dir = "."
data_dir = os.path.join(source_dir, "sample_data")
save_dir = os.path.join(source_dir, "output")
os.makedirs(save_dir, exist_ok=True)

cal_spatial_interaction = True
cal_spatial_distance = True
cal_spatial_neigh = True
num_clusters = 12 # define the number of clusters for kmeans
radius = 50 # define the radius for spatial interaction


adata = ad.read(os.path.join(data_dir, f"codex_sample_data.h5ad"))

if cal_spatial_interaction:
    adata = spatial_interaction_count(adata, 
                                      method='radius', 
                                      radius=radius, 
                                      label='spatial_interaction_count')
    adata = sm.tl.spatial_interaction(adata, 
                                    method='radius', 
                                    radius=radius, 
                                    label='spatial_interaction_radius')
    
if cal_spatial_distance:    
    adata = sm.tl.spatial_distance(adata)

if cal_spatial_neigh:
    adata = sm.tl.spatial_count(adata, phenotype='phenotype', method='radius', radius=radius, label='neigh_count')
    adata = sm.tl.spatial_cluster(adata, df_name='neigh_count', method='kmeans', k=num_clusters, label='neigh_count_kmeans')
    adata.obs['neigh_count_kmeans'].value_counts()

adata.write(os.path.join(save_dir, f"codex_processed_data.h5ad"))









