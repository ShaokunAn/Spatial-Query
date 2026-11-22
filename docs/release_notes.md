# Release Notes

## v0.0.8

- Add `motif_radius_tracking` (`spatial_query` and `spatial_query_multi`): runs de novo motif discovery at several radii, tests every discovered motif at every radius with the targeted test, and classifies how each reported motif changes at the next radius (persists, absorbed into a larger motif, splits into smaller motifs, or drops out)
- Add `plot_motif_radius_tracking` to plot these changes next to the targeted observed/expected ratios

## v0.0.2

- Add `interactive_motif` function for Vitessce visualization
- Add option to normalize spatial coordinates (`if_normalize_spatial_coord`)
- Unify adjusted p-value column name to `adj-pval` across all modules
- Return mean frequency values of each motif in differential motif analysis

## v0.0.1

- Initial release
- Single FOV spatial query (`spatial_query`)
- Multi-FOV spatial query (`spatial_query_multi`)
- Pattern mining with FP-Growth algorithm
- Motif enrichment analysis
- Differential expression analysis
