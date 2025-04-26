# Analyze CZI_kidney data

import numpy as np
import pandas as pd
import scanpy as sc
import os
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
from SpatialQuery import spatial_query
from tqdm import tqdm

plt.rcParams['pdf.fonttype']=42


file_path = '/Users/sa3520/BWH/spatial query/python/data/CZI_kidney'
files = os.listdir(file_path)
files = [f for f in files if f.endswith('.h5ad')]
data_path = os.path.join(file_path, files[0])
adata = ad.read_h5ad(data_path)

adata.var_names = adata.var['feature_name']
adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')

spatial_key = 'X_spatial'
label_key = 'predicted_label'

adatas = [ad.read_h5ad(os.path.join(file_path, f)) for f in files]