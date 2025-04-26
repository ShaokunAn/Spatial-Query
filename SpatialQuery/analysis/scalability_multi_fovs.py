# Scalability analysis on single FOV data
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import anndata as ad
from SpatialQuery import spatial_query_multi
from time import time

plt.rcParams['pdf.fonttype']=42

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)  # For pandas < 1.0 use -1 instead of None

result_path = '/Users/sa3520/BWH/spatial query/python/results/CZI_kidney/'

file_path = '/Users/sa3520/BWH/spatial query/python/data/CZI_kidney'
files = os.listdir(file_path)
files = [f for f in files if f.endswith('.h5ad')]

adatas = [ad.read_h5ad(os.path.join(file_path, f)) for f in files]

spatial_key = 'X_spatial'
label_key = 'cell_type'
feature_name = 'feature_name'
dataset_key = 'disease'

# adatas = adatas[:20]
datasets = [adata.obs[dataset_key].unique()[0] for adata in adatas]

start = time()
sps = spatial_query_multi(
    adatas=adatas,
    datasets=datasets,
    spatial_key=spatial_key,
    label_key=label_key,
    leaf_size=10,
    build_gene_index=False,
)
end = time()
print(f"Time for building index: {end - start:.2f} seconds")

unique_cts = set([l for sp in sps.spatial_queries for l in sp.labels.unique()])

ct_count_dict = dict()
for ct in unique_cts:
    ct_count_dict[ct] = [sum([len(np.where(sp.labels == ct)[0]) for sp in sps.spatial_queries])]

sorted_ct = ['macula densa epithelial cell',
             'kidney collecting duct intercalated cell',
             'kidney collecting duct principal cell',
             'kidney loop of Henle thick ascending limb epithelial cell',
             'kidney proximal convoluted tubule epithelial cell',
             ]

max_dist = 100
min_support = 0.5
print('test scalability for find_fp_dist with max_dist=100 and min_support=0.5')

for ct in sorted_ct:
    start = time()
    tt = sps.find_fp_dist(
        ct=ct,
        max_dist=max_dist,
        min_support=min_support
    )
    end = time()
    print(f"time for {ct}: {end - start:.2f} seconds")


# test scalability for motif_enrichment_dist
motif = tt['itemsets'][0]
print('test scalability for motif_enrichment_dist with max_dist=100 and min_support=0.5')

for ct in sorted_ct:
    start = time()
    tt = sps.motif_enrichment_dist(
        ct=ct,
        motifs=motif,
        max_dist=max_dist,
        min_support=min_support
    )
    end = time()
    print(f"time for {ct}: {end - start:.2f} seconds")

print('Done!')


print('test scalability for motif_enrichment_dist with max_dist=100 and min_support=0.5')

for ct in sorted_ct:
    start = time()
    tt = sps.differential_analysis_dist(
        ct=ct,
        datasets=['normal', 'diabetic kidney disease'],
    )
    end = time()
    print(f"time for {ct}: {end - start:.2f} seconds")


print('Done!')
