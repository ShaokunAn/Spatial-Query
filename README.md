# SpatialQuery

SpatialQuery is a Python package for systematic spatial analysis of single-cell resolution spatial omics data. It provides a unified framework to discover, quantify, and compare recurring spatial cell type patterns — termed *motifs* — across single or multiple fields of view (FOVs).

Starting from annotated spatial omics datasets (spatial transcriptomics and spatial proteomics data), SpatialQuery constructs cell neighborhoods using KNN or distance-based approaches and mines frequent cell type co-occurrence patterns via the FP-Growth algorithm. It then tests whether these motifs are statistically enriched beyond what is expected by chance, and supports comparison of motif compositions across biological conditions (e.g., healthy vs. disease) through differential motif analysis.

Beyond spatial structure, SpatialQuery links motifs to molecular phenotypes. For cells participating in a given motif, it performs motif-associated differential expression analysis to identify genes whose expression differs between motif-positive and motif-negative cells or across conditions. It further detects cross-cell gene-gene covariation — spatially dependent correlations between gene expression in anchor cells and their neighbors — to reveal intercellular signaling relationships that are specific to particular spatial contexts.

Key capabilities include:

- **Spatial motif discovery**: Identify frequent cell type patterns in local neighborhoods
- **Motif enrichment analysis**: Statistically test whether motifs occur more than expected
- **Differential motif analysis**: Compare spatial compositions across conditions
- **Motif-associated differential expression**: Find DE genes linked to specific motifs
- **Cross-cell gene-gene covariation**: Detect spatially dependent intercellular gene correlations
- **Multi-FOV support**: Pool and compare results across multiple tissue sections or samples

## Installation

```bash
pip install SpatialQuery
```

> **Note:** Installation typically takes ~5 minutes depending on your environment.

## Documentation

Full documentation, tutorials, and API reference: **[https://spatialquery.readthedocs.io/en/latest/](https://spatialquery.readthedocs.io/en/latest/)**

## Reproducibility

The following tutorials reproduce the main analyses presented in our manuscript. Each tutorial takes approximately 5–20 minutes to run depending on dataset size and hardware.

- [Mouse organogenesis — single-FOV analysis (seqFISH)](https://spatialquery.readthedocs.io/en/latest/tutorials/tutorial_1.html)
- [Kidney disease atlas — multi-condition analysis (MERFISH)](https://spatialquery.readthedocs.io/en/latest/tutorials/tutorial_2.html)
- [Colorectal cancer microenvironment — spatial proteomics (CODEX)](https://spatialquery.readthedocs.io/en/latest/tutorials/tutorial_3.html)
- [Whole-brain atlas — large-scale analysis (MERFISH)](https://spatialquery.readthedocs.io/en/latest/tutorials/tutorial_4.html)

## Quick Start

```python
from SpatialQuery import spatial_query

# Single FOV analysis
sq = spatial_query(adata, spatial_key="X_spatial", label_key="cell_type", feature_name="gene")
enrich_motif = sq.motif_enrichment_dist(ct="T_cell", max_dist=10, min_support=0.5)
```

```python
from SpatialQuery import spatial_query_multi

# Multi-FOV analysis
spm = spatial_query_multi(adatas=adatas, datasets=datasets,
                          spatial_key="X_spatial", label_key="cell_type", feature_name="gene")
enrich_motif = spm.motif_enrichment_dist(ct="T_cell", dataset="healthy", max_dist=10, min_support=0.5)
```

## Citation

If you use SpatialQuery in your research, please cite:

> An, S. et al. [SpatialQuery: scalable discovery and molecular characterization of multicellular motifs from spatial omics data](https://www.biorxiv.org/content/10.64898/2026.04.22.720136v1). *bioRxiv* (2026).

## License

MIT

## Contact

- **Author**: Shaokun An
- **Email**: shaokunan1@gmail.com
- **GitHub**: [ShaokunAn](https://github.com/ShaokunAn)
