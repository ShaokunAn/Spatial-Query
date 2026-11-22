# Visualization

```{toctree}
:hidden:

../generated/SpatialQuery.interactive_motif
../generated/SpatialQuery.plot_motif_composition
../generated/SpatialQuery.plot_motif_radius_tracking
../generated/SpatialQuery.maximal_patterns
../generated/SpatialQuery.retrieve_niche_pattern_freq
../generated/SpatialQuery.plot_niche_pattern_freq
```

**Interactive**

Interactive exploration of spatial motifs using [Vitessce](https://vitessce.io/).

| Method | Description |
|--------|-------------|
| {func}`~SpatialQuery.spatial_query.interactive_motif` | Launch an interactive Vitessce widget to explore spatial motifs |

**Plotting functions**

| Function | Description |
|----------|-------------|
| {doc}`SpatialQuery.plot_motif_composition <../generated/SpatialQuery.plot_motif_composition>` | Plot motif composition histograms from a composition DataFrame |
| {doc}`SpatialQuery.plot_motif_radius_tracking <../generated/SpatialQuery.plot_motif_radius_tracking>` | Plot de novo motif changes across radii from a motif_radius_tracking result |

**Utility functions**

| Function | Description |
|----------|-------------|
| {doc}`SpatialQuery.maximal_patterns <../generated/SpatialQuery.maximal_patterns>` | Keep only the maximal patterns of a frequent pattern table |
| {doc}`SpatialQuery.retrieve_niche_pattern_freq <../generated/SpatialQuery.retrieve_niche_pattern_freq>` | Frequency of each cell type of a pattern around a center cell type |
| {doc}`SpatialQuery.plot_niche_pattern_freq <../generated/SpatialQuery.plot_niche_pattern_freq>` | Heatmap of the output of retrieve_niche_pattern_freq |
