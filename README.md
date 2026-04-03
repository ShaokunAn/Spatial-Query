# SpatialQuery

A Python package for spatial query and analysis of spatial transcriptomics data, including spatial motif discovery, motif enrichment analysis, motif-associated differential expression, and cross-cell gene-gene covariation analysis.

## Installation

```bash
pip install SpatialQuery
```

## Documentation

Full documentation, tutorials, and API reference:
**[https://spatial-query.readthedocs.io](https://spatial-query.readthedocs.io)**

## Quick Start

```python
from SpatialQuery import spatial_query

# Single FOV analysis
sq = spatial_query(adata, spatial_key="X_spatial", label_key="cell_type")
fp = sq.find_fp_knn(ct="T_cell", k=30, min_support=0.5)
```

```python
from SpatialQuery import spatial_query_multi

# Multi-FOV analysis
spm = spatial_query_multi(adatas=adatas, datasets=datasets,
                          spatial_key="X_spatial", label_key="cell_type")
fp = spm.find_fp_knn(ct="T_cell", dataset="healthy", k=30, min_support=0.5)
```

## License

MIT

## Contact

- **Author**: Shaokun An
- **Email**: shaokunan1@gmail.com
- **GitHub**: [ShaokunAn](https://github.com/ShaokunAn)
