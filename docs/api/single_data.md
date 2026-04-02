# Single-data: spatial_query

```{toctree}
:hidden:

../generated/SpatialQuery.spatial_query.__init__
../generated/SpatialQuery.spatial_query.find_fp_knn
../generated/SpatialQuery.spatial_query.find_fp_dist
../generated/SpatialQuery.spatial_query.find_patterns_grid
../generated/SpatialQuery.spatial_query.find_patterns_rand
../generated/SpatialQuery.spatial_query.motif_enrichment_knn
../generated/SpatialQuery.spatial_query.motif_enrichment_dist
../generated/SpatialQuery.spatial_query.de_genes
../generated/SpatialQuery.spatial_query.compute_gene_gene_correlation
../generated/SpatialQuery.spatial_query.compute_gene_gene_correlation_by_type
../generated/SpatialQuery.spatial_query.test_score_difference
../generated/SpatialQuery.spatial_query.plot_fov
../generated/SpatialQuery.spatial_query.plot_motif_grid
../generated/SpatialQuery.spatial_query.plot_motif_rand
../generated/SpatialQuery.spatial_query.plot_motif_celltype
../generated/SpatialQuery.spatial_query.plot_all_center_motif
../generated/SpatialQuery.spatial_query.plot_fp_heatmap
../generated/SpatialQuery.spatial_query.plot_motif_enrichment_heatmap
../generated/SpatialQuery.spatial_query.plot_gene_pair_heatmap
../generated/SpatialQuery.spatial_query.plot_gene_pair_spatial
```

```python
from SpatialQuery import spatial_query
sp = spatial_query(adata, dataset="my_dataset")
```

**Motif Discovery**

| Method | Description |
|--------|-------------|
| {doc}`sp.find_fp_knn <../generated/SpatialQuery.spatial_query.find_fp_knn>` | Find frequent patterns using KNN neighborhoods |
| {doc}`sp.find_fp_dist <../generated/SpatialQuery.spatial_query.find_fp_dist>` | Find frequent patterns using distance-based neighborhoods |
| {doc}`sp.find_patterns_grid <../generated/SpatialQuery.spatial_query.find_patterns_grid>` | Find patterns on a grid |
| {doc}`sp.find_patterns_rand <../generated/SpatialQuery.spatial_query.find_patterns_rand>` | Find patterns with random sampling |
| {doc}`sp.motif_enrichment_knn <../generated/SpatialQuery.spatial_query.motif_enrichment_knn>` | Motif enrichment analysis using KNN |
| {doc}`sp.motif_enrichment_dist <../generated/SpatialQuery.spatial_query.motif_enrichment_dist>` | Motif enrichment analysis using distance |

**Motif-associated Molecular Analysis**

| Method | Description |
|--------|-------------|
| {doc}`sp.de_genes <../generated/SpatialQuery.spatial_query.de_genes>` | Differential expression analysis within motifs |
| {doc}`sp.compute_gene_gene_correlation <../generated/SpatialQuery.spatial_query.compute_gene_gene_correlation>` | Gene-gene covariation analysis |
| {doc}`sp.compute_gene_gene_correlation_by_type <../generated/SpatialQuery.spatial_query.compute_gene_gene_correlation_by_type>` | Gene-gene covariation analysis by cell type |
| {doc}`sp.test_score_difference <../generated/SpatialQuery.spatial_query.test_score_difference>` | Test covariation score differences |

**Plotting**

| Method | Description |
|--------|-------------|
| {doc}`sp.plot_fov <../generated/SpatialQuery.spatial_query.plot_fov>` | Plot field of view |
| {doc}`sp.plot_motif_grid <../generated/SpatialQuery.spatial_query.plot_motif_grid>` | Plot motifs on grid |
| {doc}`sp.plot_motif_rand <../generated/SpatialQuery.spatial_query.plot_motif_rand>` | Plot motifs with random sampling |
| {doc}`sp.plot_motif_celltype <../generated/SpatialQuery.spatial_query.plot_motif_celltype>` | Plot motifs by cell type |
| {doc}`sp.plot_all_center_motif <../generated/SpatialQuery.spatial_query.plot_all_center_motif>` | Plot all center motifs |
| {doc}`sp.plot_fp_heatmap <../generated/SpatialQuery.spatial_query.plot_fp_heatmap>` | Plot frequent pattern heatmap |
| {doc}`sp.plot_motif_enrichment_heatmap <../generated/SpatialQuery.spatial_query.plot_motif_enrichment_heatmap>` | Plot motif enrichment heatmap |
| {doc}`sp.plot_gene_pair_heatmap <../generated/SpatialQuery.spatial_query.plot_gene_pair_heatmap>` | Plot gene pair heatmap |
| {doc}`sp.plot_gene_pair_spatial <../generated/SpatialQuery.spatial_query.plot_gene_pair_spatial>` | Plot gene pair spatial expression |
