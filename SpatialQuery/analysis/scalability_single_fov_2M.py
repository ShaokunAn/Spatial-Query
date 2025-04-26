# Scalability analysis on single FOV data

import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
from SpatialQuery import spatial_query
from time import time
import os
plt.rcParams['pdf.fonttype']=42

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)  # For pandas < 1.0 use -1 instead of None

result_path = '/Users/sa3520/BWH/spatial query/python/results/CZI_kidney/'

file_path = '/Users/sa3520/BWH/spatial query/python/data/CZI_kidney'
files = os.listdir(file_path)
files = [f for f in files if f.endswith('.h5ad')]

files = files[1]

adata = ad.read_h5ad(os.path.join(file_path, files))

spatial_key = 'X_spatial'
label_key = 'cell_type'
feature_name = 'feature_name'
dataset_key = 'disease'

# adatas = adatas[:20]
dataset = adata.obs[dataset_key].unique()[0]

start = time()
sp = spatial_query(
    adata=adata,
    dataset=dataset,
    spatial_key=spatial_key,
    label_key=label_key,
    leaf_size=10,
)
end = time()
print(f"time for building index: {end - start:.2f} seconds")


ct_count_dict = adata.obs[label_key].value_counts().to_dict()

ct_ordered = [
    'kidney collecting duct intercalated cell',
    'kidney distal convoluted tubule epithelial cell',
    'leukocyte',
    'kidney proximal convoluted tubule epithelial cell'
    ]

for ct in ct_ordered:
    print(f"{ct}: {ct_count_dict[ct]} cells")

# test scalability for find_fp_dist
print('test scalability for find_fp_dist with max_dist=100 and min_support=0.5')
max_dist = 100
min_support = 0.5
for ct in ct_ordered:
    start = time()
    tt = sp.find_fp_dist(
        ct=ct,
        max_dist=max_dist,
        min_support=min_support
    )
    end = time()
    print(f"time for {ct}: {end - start:.2f} seconds")

# test scalability for motif_enrichment_dist
motif = tt['itemsets'][0]
for ct in ct_ordered:
    start = time()
    tt = sp.motif_enrichment_dist(
        ct=ct,
        motifs=motif,
        max_dist=max_dist,
        min_support=min_support
    )
    end = time()
    print(f"time for {ct}: {end - start:.2f} seconds")

print('Done!')

