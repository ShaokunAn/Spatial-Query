# SpatialQuery

**SpatialQuery** is a Python package for spatial query and analysis of spatial transcriptomics data.
It provides efficient methods to identify cell-type spatial co-occurrence patterns (motifs),
perform motif enrichment analysis, and conduct motif-associated molecular analysis
within spatial neighborhoods.

## Method Overview

```{image} _static/img/workflow.png
:alt: SpatialQuery method overview
:width: 100%
:align: center
```

## Key Applications

### Single-dataset Analysis

::::{grid} 3
:gutter: 3

:::{grid-item-card} Motif Enrichment Analysis
:link: examples/single_motif_enrichment
:link-type: doc

Identify frequent cell-type co-occurrence patterns (motifs) using FP-Growth algorithm
and evaluate their spatial enrichment via KNN or distance-based neighborhoods.
:::

:::{grid-item-card} Differential Expression Analysis
:link: examples/single_differential_expression
:link-type: doc

Detect genes differentially expressed within spatial motifs.
:::

:::{grid-item-card} Gene-Gene Covariation Analysis
:link: examples/single_gene_gene_covariation
:link-type: doc

Quantify gene-gene covariation patterns associated with specific motifs.
:::

::::

### Multi-dataset Analysis

::::{grid} 2
:gutter: 3

:::{grid-item-card} Motif Enrichment Analysis
:link: examples/multi_motif_enrichment
:link-type: doc

Perform motif enrichment analysis across multiple FOVs or tissue samples simultaneously.
:::

:::{grid-item-card} Differential Motif Analysis
:link: examples/multi_differential_motif
:link-type: doc

Compare motif frequencies across conditions to identify differentially enriched
spatial patterns between groups.
:::

:::{grid-item-card} Differential Expression Analysis
:link: examples/multi_differential_expression
:link-type: doc

Detect differentially expressed genes within spatial motifs across multiple datasets.
:::

:::{grid-item-card} Gene-Gene Covariation Analysis
:link: examples/multi_gene_gene_covariation
:link-type: doc

Quantify gene-gene covariation patterns across multiple datasets.
:::

::::

## Getting Started

::::{grid} 3
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc

How to install SpatialQuery.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Step-by-step guides for single and multi-FOV analysis.
:::

:::{grid-item-card} API Reference
:link: api
:link-type: doc

Full API documentation for all modules.
:::

::::

## Reference

If you use SpatialQuery in your research, please cite:

> *Citation placeholder — to be updated with publication details.*

```{toctree}
:hidden:
:caption: GENERAL

installation
api
```

```{toctree}
:hidden:
:caption: GALLERY

examples/index
tutorials/index
```

```{toctree}
:hidden:

release_notes
```
