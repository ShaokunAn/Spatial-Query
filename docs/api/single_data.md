# Single-data: spatial_query

```{toctree}
:hidden:

../generated/SpatialQuery.spatial_query.__init__
../generated/SpatialQuery.spatial_query.get_genes
../generated/SpatialQuery.spatial_query.get_labels
../generated/SpatialQuery.spatial_query.get_dataset
../generated/SpatialQuery.spatial_query.find_fp_knn
../generated/SpatialQuery.spatial_query.find_fp_dist
../generated/SpatialQuery.spatial_query.find_patterns_grid
../generated/SpatialQuery.spatial_query.find_patterns_rand
../generated/SpatialQuery.spatial_query.motif_enrichment_knn
../generated/SpatialQuery.spatial_query.motif_enrichment_dist
../generated/SpatialQuery.spatial_query.motif_composition_knn
../generated/SpatialQuery.spatial_query.motif_composition_dist
../generated/SpatialQuery.spatial_query.motif_radius_tracking
../generated/SpatialQuery.spatial_query.de_genes
../generated/SpatialQuery.spatial_query.get_anchor_motif_cell_ids
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
../generated/SpatialQuery.spatial_query.plot_motif_composition
../generated/SpatialQuery.spatial_query.plot_motif_radius_tracking
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
| {doc}`sp.motif_composition_knn <../generated/SpatialQuery.spatial_query.motif_composition_knn>` | Per-neighborhood cell type composition of a motif using KNN |
| {doc}`sp.motif_composition_dist <../generated/SpatialQuery.spatial_query.motif_composition_dist>` | Per-neighborhood cell type composition of a motif using distance |
| {doc}`sp.motif_radius_tracking <../generated/SpatialQuery.spatial_query.motif_radius_tracking>` | Follow de novo motifs across radii, with a targeted test of each at every radius |

**Motif-associated Molecular Analysis**

| Method | Description |
|--------|-------------|
| {doc}`sp.de_genes <../generated/SpatialQuery.spatial_query.de_genes>` | Differential expression analysis within motifs |
| {doc}`sp.compute_gene_gene_correlation <../generated/SpatialQuery.spatial_query.compute_gene_gene_correlation>` | Gene-gene covariation analysis |
| {doc}`sp.compute_gene_gene_correlation_by_type <../generated/SpatialQuery.spatial_query.compute_gene_gene_correlation_by_type>` | Gene-gene covariation analysis by cell type |
| {doc}`sp.test_score_difference <../generated/SpatialQuery.spatial_query.test_score_difference>` | Test covariation score differences |
| {doc}`sp.get_anchor_motif_cell_ids <../generated/SpatialQuery.spatial_query.get_anchor_motif_cell_ids>` | Cell indices of motif-positive anchors and their motif neighbors |

**Data Access**

| Method | Description |
|--------|-------------|
| {doc}`sp.get_genes <../generated/SpatialQuery.spatial_query.get_genes>` | Gene names of the dataset |
| {doc}`sp.get_labels <../generated/SpatialQuery.spatial_query.get_labels>` | Cell type label of every cell |
| {doc}`sp.get_dataset <../generated/SpatialQuery.spatial_query.get_dataset>` | Dataset name |

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
| {doc}`sp.plot_motif_composition <../generated/SpatialQuery.spatial_query.plot_motif_composition>` | Plot motif composition histograms with median and IQR |
| {doc}`sp.plot_motif_radius_tracking <../generated/SpatialQuery.spatial_query.plot_motif_radius_tracking>` | Plot de novo motif changes across radii next to targeted O/E |
| {doc}`sp.plot_gene_pair_heatmap <../generated/SpatialQuery.spatial_query.plot_gene_pair_heatmap>` | Plot gene pair heatmap |
| {doc}`sp.plot_gene_pair_spatial <../generated/SpatialQuery.spatial_query.plot_gene_pair_spatial>` | Plot gene pair spatial expression |
