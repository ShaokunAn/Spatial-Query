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
pip install -e .

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
    build_gene_index=True,  # Build gene expression index for DE analysis
    feature_name="gene_ids"  # gene names in adata.var
)

# Find frequent patterns around a specific cell type
fp_results = sq.find_fp_knn(
    ct="T_cell",  # cell type of interest
    k=30,  # number of neighbors
    min_support=0.5  # minimum support threshold
)

# Perform motif enrichment analysis
enrichment_results = sq.motif_enrichment_knn(
    motif=["T_cell", "B_cell"],  # motif to test
    ct="T_cell",  # center cell type
    k=30
)

# Differential expression analysis
de_results = sq.de_genes(
    ind_group1=[0, 1, 2, 3],  # indices of group 1 cells
    ind_group2=[4, 5, 6, 7],  # indices of group 2 cells
    method="fisher"  # Fisher's exact test
)

# Visualize results
sq.plot_fov(fig_size=(10, 8))
sq.plot_motif_grid(motif=["T_cell", "B_cell"])
```

### Multi-FOV Analysis

```python
from SpatialQuery import spatial_query_multi

# Prepare multiple datasets
adatas = [adata1, adata2, adata3]  # List of AnnData objects
datasets = ["sample1", "sample2", "sample3"]  # Dataset names

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
    dataset=["sample1", "sample2"],  # specific datasets
    k=30,
    min_support=0.5
)

# Differential analysis across datasets
diff_results = sq_multi.differential_analysis_knn(
    ct="T_cell",
    datasets1=["sample1"],
    datasets2=["sample2"],
    k=30
)

# Cell type distribution analysis
dist_results = sq_multi.cell_type_distribution()
```

## Core Classes and Methods

### `spatial_query` Class (Single FOV)

The main class for analyzing spatial patterns within a single field of view.

#### Key Methods:

- **`find_fp_knn(ct, k, min_support)`**: Find frequent patterns around a cell type using k-nearest neighbors
- **`find_fp_dist(ct, max_dist, min_support)`**: Find frequent patterns using distance-based neighborhoods
- **`motif_enrichment_knn(motif, ct, k)`**: Test motif enrichment using k-NN neighborhoods
- **`motif_enrichment_dist(motif, ct, max_dist)`**: Test motif enrichment using distance-based neighborhoods
- **`find_patterns_grid(max_dist, min_support)`**: Find patterns using grid-based sampling
- **`find_patterns_rand(max_dist, n_points, min_support)`**: Find patterns using random sampling
- **`de_genes(ind_group1, ind_group2, method)`**: Differential expression analysis
- **`plot_fov(fig_size)`**: Visualize the spatial data
- **`plot_motif_grid(motif, max_dist)`**: Plot motif distribution around grid points
- **`plot_motif_rand(motif, max_dist, n_points)`**: Plot motif distribution around random points
- **`plot_motif_celltype(motif, ct, max_dist)`**: Plot motif around specific cell types

#### Parameters:
- `adata`: AnnData object containing spatial transcriptomics data
- `dataset`: Dataset name (default: 'ST')
- `spatial_key`: Key for spatial coordinates in `adata.obsm` (default: 'X_spatial')
- `label_key`: Key for cell type labels in `adata.obs` (default: 'predicted_label')
- `build_gene_index`: Whether to build gene expression index (default: False)
- `feature_name`: Gene names key in `adata.var` (required if `build_gene_index=True`)

### `spatial_query_multi` Class (Multi-FOV)

The main class for analyzing spatial patterns across multiple fields of view or datasets.

#### Key Methods:

- **`find_fp_knn(ct, dataset, k, min_support)`**: Find frequent patterns across specified datasets
- **`find_fp_dist(ct, dataset, max_dist, min_support)`**: Find patterns using distance-based neighborhoods
- **`motif_enrichment_knn(motif, ct, dataset, k)`**: Test motif enrichment across datasets
- **`motif_enrichment_dist(motif, ct, dataset, max_dist)`**: Distance-based motif enrichment
- **`find_fp_knn_fov(ct, fov, k, min_support)`**: Find patterns in specific FOV
- **`find_fp_dist_fov(ct, fov, max_dist, min_support)`**: Distance-based patterns in specific FOV
- **`differential_analysis_knn(ct, datasets1, datasets2, k)`**: Compare patterns between dataset groups
- **`differential_analysis_dist(ct, datasets1, datasets2, max_dist)`**: Distance-based differential analysis
- **`de_genes(ind_group1, ind_group2, dataset, method)`**: Differential expression analysis
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

# Create example data
n_cells = 1000
n_genes = 2000

# Spatial coordinates
spatial_coords = np.random.rand(n_cells, 2) * 100

# Cell type labels
cell_types = np.random.choice(['T_cell', 'B_cell', 'Macrophage', 'Neuron'], n_cells)

# Gene expression matrix
expression_matrix = np.random.negative_binomial(5, 0.3, (n_cells, n_genes))

# Create AnnData object
adata = sc.AnnData(X=expression_matrix)
adata.obsm['X_spatial'] = spatial_coords
adata.obs['predicted_label'] = cell_types
adata.var['gene_ids'] = [f'Gene_{i}' for i in range(n_genes)]

# Optional: Add gene names as index
adata.var_names = adata.var['gene_ids']
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
    motif=["T_cell", "B_cell", "Macrophage"],
    ct="T_cell",
    k=30
)
```

### Performance Optimization

For large datasets, consider:

```python
# Use distance-based methods for better performance
fp_results = sq.find_fp_dist(
    ct="T_cell",
    max_dist=100,  # distance threshold
    min_support=0.5
)

# Build gene index only when needed
sq = spatial_query(adata, build_gene_index=False)  # Faster initialization
# Later: sq.build_gene_index() if DE analysis needed
```

## Citation

If you use Spatial-Query in your research, please cite:

```bibtex
@software{spatial_query,
  title={Spatial-Query: Fast spatial query and analysis for spatial transcriptomics},
  author={Shaokun An},
  year={2024},
  url={https://github.com/ShaokunAn/Spatial-Query}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Shaokun An
- **Email**: shan12@bwh.harvard.edu
- **GitHub**: [@ShaokunAn](https://github.com/ShaokunAn)

## Acknowledgments

This package builds upon several excellent open-source libraries including scanpy, scikit-learn, and mlxtend.