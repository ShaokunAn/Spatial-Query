# Spatial-Query

A Python package for fast spatial query and analysis of Spatial Transcriptomics (ST) data. Spatial-Query provides efficient methods to identify frequent patterns, perform motif enrichment analysis, and conduct differential expression analysis in spatial transcriptomics datasets.

## Features

- **Single FOV Analysis**: Analyze spatial patterns within individual fields of view
- **Multi-FOV Analysis**: Compare patterns across multiple fields of view or datasets
- **Fast Spatial Queries**: Built on k-D tree for efficient spatial neighborhood queries
- **Pattern Mining**: Identify frequent cell type patterns using FP-Growth algorithm
- **Motif Enrichment**: Statistical analysis of spatial motif enrichment
- **Differential Expression**: Gene expression analysis with Fisher's exact test
- **Visualization**: Comprehensive plotting functions for spatial data

## Installation

### From GitHub Repository

```bash
# Clone the repository
git clone https://github.com/ShaokunAn/Spatial-Query.git
cd Spatial-Query

# Install in development mode
pip install .

# Or install directly from GitHub
pip install git+https://github.com/ShaokunAn/Spatial-Query.git@dev
```

### Dependencies

The package requires the following dependencies:
- Python >= 3.8
- numpy, pandas, scipy
- matplotlib, seaborn
- scikit-learn
- scanpy, anndata
- mlxtend
- statsmodels
- pybind11 (for C++ extensions)

## Quick Start

### Single FOV Analysis

```python
import scanpy as sc
from SpatialQuery import spatial_query

# Load your spatial transcriptomics data
adata = sc.read_h5ad("your_data.h5ad")

# Initialize spatial query object
sq = spatial_query(
    adata=adata,
    dataset="ST_sample",
    spatial_key="X_spatial",  # spatial coordinates in adata.obsm
    label_key="predicted_label",  # cell type labels in adata.obs
    build_gene_index=False,  # build gene expression index. If set True, build scfind index otherwise use adata.X directly for DE gene analysis
    feature_name="gene_ids",  # gene names in adata.var
    if_lognorm=True  # perfrom log-normalization of adata.X if True when initializing spatial_query object. 

)

# Find frequent patterns around a specific cell type
fp_results = sq.find_fp_knn(
    ct="T_cell",  # anchors cells for neighborhood analysis
    k=30,  # number of neighbors
    min_support=0.5  # minimum frequency support threshold
)

# Perform motif enrichment analysis
enrichment_results = sq.motif_enrichment_knn(
    ct="T_cell",  # center cell type as anchors
    motifs=["T_cell", "B_cell"],  # motif to test. If None, frequent patterns will be searched first for enrichment analysis
    k=30,  # number of neighbors
    min_support=0.5,  # minimum frequency support threshold
    max_dist=200,  # maximum distance for neighbors
    return_cellID=False  # whether to return cell IDs for each motif and center cells
)

# Differential expression analysis
de_results = sq.de_genes(
    ind_group1=[0, 1, 2, 3],  # indices of group 1 cells
    ind_group2=[4, 5, 6, 7],  # indices of group 2 cells
    method="fisher"  # Fisher's exact test
)

# Visualize results
sq.plot_fov(fig_size=(10, 8))  # Plot spatial data with cell types
sq.plot_motif_grid(motif=["T_cell", "B_cell"], max_dist=50)  # Plot motif around grid points
sq.plot_motif_celltype(
    ct="T_cell",  # center cell type
    motif=["T_cell", "B_cell"],  # motif to visualize
    max_dist=100,  # radius for neighborhood
    fig_size=(10, 5),
    save_path=None  # path to save figure, None for display only
)
```

### Multi-FOV Analysis

```python
from SpatialQuery import spatial_query_multi

# Prepare multiple datasets
adatas = [adata1, adata2, adata3]  # List of AnnData objects
datasets = ["healthy", "healthy", "disease"]  # Dataset names. 

# Initialize multi-FOV spatial query
sq_multi = spatial_query_multi(
    adatas=adatas,
    datasets=datasets,
    spatial_key="X_spatial",
    label_key="predicted_label",
    build_gene_index=True,
    feature_name="gene_ids"
)

# Find frequent patterns across datasets
fp_multi = sq_multi.find_fp_knn(
    ct="T_cell",
    dataset=["healthy"],  # specific datasets
    k=30,
    min_support=0.5
)

# Motif enrichment analysis across datasets
motif_results = sq_multi.motif_enrichment_knn(
    ct="T_cell",  # center cell type
    motifs=["T_cell", "B_cell"],  # motifs to test
    dataset=["healthy", "disease"],  # datasets to compare
    k=30,
    min_support=0.5,
    max_dist=200
)

# Differential pattern analysis across datasets
diff_results = sq_multi.differential_analysis_knn(
    ct="T_cell",  # center cell type
    datasets=["healthy", "disease"],  # exactly 2 datasets for comparison
    k=30,  # number of neighbors
    min_support=0.5,  # minimum support threshold
    max_dist=200  # maximum distance for neighbors
)

# Differential gene expression analysis across specified groups using per-dataset indices
from collections import defaultdict

# Example: keys are modified dataset names (e.g., "healthy_0", "healthy_1"), values are index lists for that dataset
ind_group1 = defaultdict(list)
ind_group1["healthy_0"] = [0, 1, 2]
ind_group1["healthy_1"] = [0, 1]

ind_group2 = defaultdict(list)
ind_group2["disease_0"] = [3, 4]


de_multi = sq_multi.de_genes(
    ind_group1=ind_group1,  # group 1: dict keys as dataset names, values as indices in each dataset
    ind_group2=ind_group2,  # group 2: same structure
    genes=["Gene_1", "Gene_2"],      # Genes of interest; uses all genes if no genes are input
    method="fisher"         # method to perform differential gene analysis
)

# Cell type distribution analysis across datasets
dist_results = sq_multi.cell_type_distribution()  # overall distribution
dist_fov = sq_multi.cell_type_distribution_fov()  # per-FOV distribution

# Visualize results for each FOV
for i, sq in enumerate(sq_multi.spatial_queries):
    sq.plot_fov(fig_size=(8, 6))
    sq.plot_motif_celltype(
        ct="T_cell",
        motif=["T_cell", "B_cell"],
        max_dist=50
    )
```

## Core Classes and Methods

### `spatial_query` Class (Single FOV)

The main class for analyzing spatial patterns within a single field of view.

#### Key Methods:

- **`find_fp_knn(ct, k, min_support)`**: Find frequent patterns around a cell type using k-nearest neighbors
- **`find_fp_dist(ct, max_dist, min_support)`**: Find frequent patterns using distance-based neighborhoods
- **`motif_enrichment_knn(ct, motifs, k, min_support, max_dist)`**: Test motif enrichment using k-NN neighborhoods
- **`motif_enrichment_dist(ct, motifs, max_dist, min_support)`**: Test motif enrichment using distance-based neighborhoods
- **`find_patterns_grid(max_dist, min_support)`**: Find patterns using grid-based sampling
- **`find_patterns_rand(max_dist, n_points, min_support)`**: Find patterns using random sampling
- **`de_genes(ind_group1, ind_group2, method)`**: Differential expression analysis
- **`plot_fov(fig_size)`**: Visualize the spatial data
- **`plot_motif_grid(motif, max_dist)`**: Plot motif distribution around grid points
- **`plot_motif_rand(motif, max_dist, n_points)`**: Plot motif distribution around random sampled points
- **`plot_motif_celltype(motif, ct, max_dist)`**: Plot motif around specific cell types

#### Parameters:
- `adata`: AnnData object containing spatial transcriptomics data
- `dataset`: Dataset name (default: 'ST')
- `spatial_key`: Key for spatial coordinates in `adata.obsm` (default: 'X_spatial')
- `label_key`: Key for cell type labels in `adata.obs` (default: 'predicted_label')
- `build_gene_index`: Whether to build gene expression index with scfind (default: False)
- `feature_name`: Gene names key in `adata.var` (required if `build_gene_index=True`)

### `spatial_query_multi` Class (Multi-FOV)

The main class for analyzing spatial patterns across multiple fields of view or datasets.

#### Key Methods:

- **`find_fp_knn(ct, dataset, k, min_support)`**: Find frequent patterns across specified datasets
- **`find_fp_dist(ct, dataset, max_dist, min_support)`**: Find patterns using distance-based neighborhoods
- **`motif_enrichment_knn(ct, motifs, dataset, k, min_support, max_dist)`**: Test motif enrichment across datasets
- **`motif_enrichment_dist(ct, motifs, dataset, max_dist, min_support)`**: Distance-based motif enrichment
- **`differential_analysis_knn(ct, datasets, k, min_support, max_dist)`**: Compare patterns between dataset groups
- **`differential_analysis_dist(ct, datasets, max_dist, min_support)`**: Distance-based differential pattern analysis
- **`de_genes(ind_group1, ind_group2, gene, method)`**: Differential expression analysis
- **`cell_type_distribution()`**: Analyze cell type distribution across datasets
- **`cell_type_distribution_fov()`**: Cell type distribution per FOV

#### Parameters:
- `adatas`: List of AnnData objects
- `datasets`: List of dataset names
- `spatial_key`: Key for spatial coordinates
- `label_key`: Key for cell type labels
- `build_gene_index`: Whether to build gene expression indices

## Data Format Requirements

### AnnData Object Structure

Your AnnData object should contain:

- **`adata.obsm['X_spatial']`**: Spatial coordinates (n_cells × 2)
- **`adata.obs['predicted_label']`**: Cell type labels
- **`adata.var['gene_ids']`**: Gene names (if using gene expression analysis)
- **`adata.X`**: Gene expression matrix (if using gene expression analysis)

### Example Data Preparation

```python
import scanpy as sc
import pandas as pd
import numpy as np

# Create example spatial transcriptomics data
n_cells = 1000
n_genes = 2000

# Spatial coordinates (2D coordinates for each cell)
spatial_coords = np.random.rand(n_cells, 2) * 100

# Cell type labels (annotated cell types)
cell_types = np.random.choice(['T_cell', 'B_cell', 'Macrophage', 'Neuron'], n_cells)

# Gene expression matrix (cells × genes)
expression_matrix = np.random.negative_binomial(5, 0.3, (n_cells, n_genes))

# Create AnnData object
adata = sc.AnnData(X=expression_matrix)
adata.obsm['X_spatial'] = spatial_coords  # Required: spatial coordinates
adata.obs['predicted_label'] = cell_types  # Required: cell type labels
adata.var['gene_ids'] = [f'Gene_{i}' for i in range(n_genes)]  # Required for gene analysis

# Optional: Add gene names as index
adata.var_names = adata.var['gene_ids']

# Optional: Add metadata
adata.obs['sample_id'] = ['sample_1'] * n_cells
adata.obs['region'] = np.random.choice(['cortex', 'medulla'], n_cells)
```

### Loading Real Data

```python
# Load from common spatial transcriptomics formats
import scanpy as sc

# Load 10X Visium data
adata = sc.read_10x_h5("filtered_feature_bc_matrix.h5")
adata.var_names_unique()

# Load spatial coordinates (from spaceranger output)
spatial_coords = pd.read_csv("spatial/tissue_positions_list.csv", 
                            header=None, index_col=0)
spatial_coords = spatial_coords[[1, 2]].values  # x, y coordinates
adata.obsm['X_spatial'] = spatial_coords

# Load cell type annotations (from external analysis)
cell_types = pd.read_csv("cell_type_annotations.csv")
adata.obs['predicted_label'] = cell_types['cell_type'].values

# Initialize spatial query
sq = spatial_query(adata, build_gene_index=False, feature_name="gene_ids")
```

## Advanced Usage

### Custom Spatial Analysis

```python
# Custom neighborhood analysis
sq = spatial_query(adata, build_gene_index=True)

# Find patterns with custom parameters
fp_results = sq.find_fp_knn(
    ct="T_cell",
    k=50,  # larger neighborhood
    min_support=0.3  # lower support threshold
)

# Test specific motifs
motif_results = sq.motif_enrichment_knn(
    ct="T_cell",
    motifs=["T_cell", "B_cell", "Macrophage"],
    k=30,
    min_support=0.5,
    max_dist=200
)
```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Shaokun An
- **Email**: shaokunan1@gmail.com
- **GitHub**: [@ShaokunAn](https://github.com/ShaokunAn)

## Acknowledgments

This package builds upon several excellent open-source libraries including scanpy, scikit-learn, and mlxtend.