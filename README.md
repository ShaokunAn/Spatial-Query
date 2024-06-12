SpatialQuery is a package for fast query of Spatial Transcriptomics data. 

### Analysis of ST data in SpatialQuery

With annotated ST data as input, SpatialQuery first builds a k-D tree based on spatial location in each FOV for fast query of neighboring cell compositions. It is composed of methods for single-FOV and multiple-FOVs.
In single-FOV, it contains methods:

- identify frequent patterns across FOV ()
- identify frequent patterns around cell type of interest
- identify statistically significant patterns around cell type of interest

For multiple-FOVs data, it contains methods:

- identify frequent patterns around cell type of interest in specified dataset
- identify statistically significant patterns around cell type of interest in specified dataset
- identify differential patterns across datasets

### Installation

```
pip install SpatialQuery
```





