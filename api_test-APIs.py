# test SpatialQuery API
from time import time

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

data_file='/Users/sa3520/BWH/spatial query/python/data/secondary_analysis.h5ad'

adata = ad.read_h5ad(data_path)

spatial_pos = adata.obsm['X_spatial']
labels = adata.obs['predicted_label']

# 发送POST请求
url = "http://3.23.17.244:8080/api/upload_spatial"  # 替换为你的实际URL
headers = {'Content-Type': 'application/json'}

data = {
    "spatial_pos": spatial_pos.tolist(),  # 将numpy数组转为列表
    "labels": labels.tolist(),
    "dataset_id": f"test_dataset_32k",
    "leaf_size": 10,
    "max_radius": 500,
    "n_split": 10
}
start = time()
try:
    response = requests.post(url, json=data, headers=headers)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")

end = time()
print(f'time: {end - start:.2f} seconds to upload {len(spatial_pos)} cells')
print(response.json())

# test api of find_fp_dist
url = "http://3.23.17.244:8080/api/find_fp_dist"
data = {
    "ct": "B cell",
    "max_dist": 100,
    "min_support": 0.5,
    "dataset_id": "test_dataset_32k"
}
headers = {'Content-Type': 'application/json'}
start = time()
try:
    response = requests.post(url, json=data, headers=headers)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")

end = time()
print(f'time: {end - start:.2f} seconds to call find_fp_dist cells')
print(response.json())
fp_dist = pd.DataFrame(response.json()['frequent_patterns'])


# test api of find_fp_knn
url = "http://3.23.17.244:8080/api/find_fp_knn"
data = {
    "ct": "B cell",
    "k": 30,
    "min_support": 0.5,
    "dataset_id": "test_dataset_32k"
}
headers = {'Content-Type': 'application/json'}
start = time()
try:
    response = requests.post(url, json=data, headers=headers)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")

end = time()
print(f'time: {end - start:.2f} seconds to call find_fp_dist cells')
print(response.json())


# test api of motif_enrichment_dist
url = "http://3.23.17.244:8080/api/motif_enrichment_knn"
data = {
    "ct": "B cell",
    "motifs": fp_dist['itemsets'][0],
    "k": 30,
    "min_support": 0.5,
    "dataset_id": "test_dataset_32k"
}
headers = {'Content-Type': 'application/json'}
start = time()
try:
    response = requests.post(url, json=data, headers=headers)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")

end = time()
print(f'time: {end - start:.2f} seconds to call find_fp_dist cells')
print(response.json())

url = "http://3.23.17.244:8080/api/motif_enrichment_knn"
data = {
    "ct": "podocyte",
    "k": 30,
    "min_support": 0.5,
    "dataset_id": "test_dataset_32k"
}
headers = {'Content-Type': 'application/json'}
start = time()
try:
    response = requests.post(url, json=data, headers=headers)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")

end = time()
print(f'time: {end - start:.2f} seconds to call find_fp_dist cells')
print(response.json())
out = pd.DataFrame(response.json()['enrichment_results'])


# test api of motif_enrichment_dist
url = "http://3.23.17.244:8080/api/motif_enrichment_dist"
data = {
    "ct": "podocyte",
    "motifs": fp_dist['itemsets'][0],
    "max_dist": 100,
    "min_support": 0.5,
    "dataset_id": "test_dataset_32k"
}
headers = {'Content-Type': 'application/json'}
start = time()
try:
    response = requests.post(url, json=data, headers=headers)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")

end = time()
print(f'time: {end - start:.2f} seconds to call find_fp_dist cells')
print(response.json())

data = {
    "ct": "podocyte",
    "max_dist": 100,
    "min_support": 0.5,
    "dataset_id": "test_dataset_32k"
}
headers = {'Content-Type': 'application/json'}
start = time()
try:
    response = requests.post(url, json=data, headers=headers)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")

end = time()
print(f'time: {end - start:.2f} seconds to call find_fp_dist cells')
print(response.json())

# test api of list_datasets
url = "http://3.23.17.244:8080/api/list_datasets"
start = time()
try:
    response = requests.get(url,)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")

end = time()
print(f'time: {end - start:.2f} seconds to list datasets')

tt1 = response.json()['available_datasets']

# test api of remoing dataset
url = "http://3.23.17.244:8080/api/remove_dataset"
data = {
    "dataset_id": "test_dataset_32k"
}
headers = {'Content-Type': 'application/json'}
start = time()
try:
    response = requests.post(url, json=data, headers=headers)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")

end = time()
print(f'time: {end - start:.2f} seconds to remove dataset')
tt2 = response.json()['available_datasets']

# All APIs work as expected!