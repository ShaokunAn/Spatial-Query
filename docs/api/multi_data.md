# Multi-data: spatial_query_multi

```{toctree}
:hidden:

../generated/SpatialQuery.spatial_query_multi.__init__
../generated/SpatialQuery.spatial_query_multi.find_fp_knn
../generated/SpatialQuery.spatial_query_multi.find_fp_dist
../generated/SpatialQuery.spatial_query_multi.motif_enrichment_knn
../generated/SpatialQuery.spatial_query_multi.motif_enrichment_dist
../generated/SpatialQuery.spatial_query_multi.differential_analysis_knn
../generated/SpatialQuery.spatial_query_multi.differential_analysis_dist
../generated/SpatialQuery.spatial_query_multi.de_genes
../generated/SpatialQuery.spatial_query_multi.compute_gene_gene_correlation
../generated/SpatialQuery.spatial_query_multi.compute_gene_gene_correlation_by_type
../generated/SpatialQuery.spatial_query_multi.test_score_difference
../generated/SpatialQuery.spatial_query_multi.plot_cell_type_distribution
../generated/SpatialQuery.spatial_query_multi.plot_cell_type_distribution_fov
```

```python
from SpatialQuery import spatial_query_multi
spm = spatial_query_multi(adatas, datasets=["dataset1", "dataset2"])
```

**Motif Discovery**

| Method | Description |
|--------|-------------|
| {doc}`spm.find_fp_knn <../generated/SpatialQuery.spatial_query_multi.find_fp_knn>` | Find frequent patterns using KNN neighborhoods |
| {doc}`spm.find_fp_dist <../generated/SpatialQuery.spatial_query_multi.find_fp_dist>` | Find frequent patterns using distance-based neighborhoods |
| {doc}`spm.motif_enrichment_knn <../generated/SpatialQuery.spatial_query_multi.motif_enrichment_knn>` | Motif enrichment analysis using KNN |
| {doc}`spm.motif_enrichment_dist <../generated/SpatialQuery.spatial_query_multi.motif_enrichment_dist>` | Motif enrichment analysis using distance |

**Differential Motif Analysis**

| Method | Description |
|--------|-------------|
| {doc}`spm.differential_analysis_knn <../generated/SpatialQuery.spatial_query_multi.differential_analysis_knn>` | Differential motif analysis using KNN |
| {doc}`spm.differential_analysis_dist <../generated/SpatialQuery.spatial_query_multi.differential_analysis_dist>` | Differential motif analysis using distance |

**Motif-associated Molecular Analysis**

| Method | Description |
|--------|-------------|
| {doc}`spm.de_genes <../generated/SpatialQuery.spatial_query_multi.de_genes>` | Differential expression analysis within motifs |
| {doc}`spm.compute_gene_gene_correlation <../generated/SpatialQuery.spatial_query_multi.compute_gene_gene_correlation>` | Gene-gene covariation analysis |
| {doc}`spm.compute_gene_gene_correlation_by_type <../generated/SpatialQuery.spatial_query_multi.compute_gene_gene_correlation_by_type>` | Gene-gene covariation analysis by cell type |
| {doc}`spm.test_score_difference <../generated/SpatialQuery.spatial_query_multi.test_score_difference>` | Test covariation score differences |

**Plotting**

| Method | Description |
|--------|-------------|
| {doc}`spm.plot_cell_type_distribution <../generated/SpatialQuery.spatial_query_multi.plot_cell_type_distribution>` | Plot cell type distribution |
| {doc}`spm.plot_cell_type_distribution_fov <../generated/SpatialQuery.spatial_query_multi.plot_cell_type_distribution_fov>` | Plot cell type distribution per FOV |
