
from typing import List, Tuple, Union, Optional, Literal

import matplotlib.pyplot as plt
import statsmodels.stats.multitest as mt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from pandas import DataFrame
from scipy.spatial import KDTree
from scipy import sparse
from scipy.stats import hypergeom, fisher_exact
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from statsmodels.stats.multitest import multipletests

from time import time

import zarr

from .scfind4sp import SCFind
import scanpy as sc
from . import spatial_utils, spatial_gene_covarying, plotting



class spatial_query:
    """
    Spatial Query Analysis for Single Field of View (FOV)

    A class for performing spatial query analysis on spatial transcriptomics data of a single field of view,
    including pattern discovery, enrichment analysis, differential analysis, and gene-gene covariation assessment.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing expression matrix and metadata.
    dataset : str, default='ST'
        Dataset identifier.
    spatial_key : str, default='X_spatial'
        Key in adata.obsm containing spatial coordinates (n_obs, 2).
    label_key : str, default='cell_type'
        Key in adata.obs containing cell type annotations.
    leaf_size : int, default=10
        Leaf size parameter for KDTree construction. Larger values reduce tree depth
        but increase computation per query.
    build_gene_index : bool, default=False
        If True, constructs SCFind index by compressing original data to binary format to save memory.
        If False, uses original expression matrix from adata.X directly.
    feature_name : str, default='gene'
        Column name in adata.var containing feature identifiers.
    if_lognorm : bool, default=True
        If True, applies log1p transformation after library size normalization.
        Set to False if expression data is already normalized and no more transformation is required.
    if_normalize_spatial_coord : bool, default=True
        If True, normalizes spatial coordinates so mean nearest neighbor distance equals 1.
        This makes max_dist parameter interpretable as "number of cell diameters."
        Set to False if original spatial units (e.g., micrometers) are required and
        should be preserved for downstream analysis.

    Notes
    -----
    When if_normalize_spatial_coord=True, the max_dist parameter in query functions
    represents relative spatial scale (in units of mean nearest neighbor distance).
    When if_normalize_spatial_coord=False, max_dist uses original coordinate units.
    """
    def __init__(
        self,
        adata: AnnData,
        dataset: str = 'ST',
        spatial_key: str = 'X_spatial',
        label_key: str = 'cell_type',
        leaf_size: int = 10,
        build_gene_index: bool = False,
        feature_name: str = 'gene',
        if_lognorm: bool = True,
        if_normalize_spatial_coord: bool = True
        ):
        if spatial_key not in adata.obsm.keys() or label_key not in adata.obs.keys():
            raise ValueError(f"The Anndata object must contain {spatial_key} in obsm and {label_key} in obs.")
        
        self.spatial_key = spatial_key
        self.feature_name = feature_name

        # Standardize spatial position
        self.spatial_pos = np.array(adata.obsm[self.spatial_key])
        if if_normalize_spatial_coord:
            self.spatial_pos = spatial_utils._auto_normalize_spatial_coords(self.spatial_pos)
        
        self.dataset = dataset
        self.label_key = label_key
        self.labels = adata.obs[self.label_key].reset_index(drop=True)
        self.labels = self.labels.astype('category')
        self.kd_tree = KDTree(self.spatial_pos, leafsize=leaf_size)
        self.build_gene_index = build_gene_index
        
        self.adata = None
        self.genes = None
        self.index = None

        # filter features with NA
        valid_features = adata.var[feature_name].isna().values
        adata = adata[:, ~valid_features]
        # filter duplicated features
        duplicated = adata.var.duplicated(subset=[feature_name], keep='first')
        adata = adata[:, ~duplicated.values].copy()
        adata.var_names = adata.var[feature_name].tolist()
        
        if build_gene_index:
            # Store data with scfind method
            if '_' not in dataset:
                # Add _ to dataset_name if missing, to keep name format consistent when dealing with
                # multiple FOVs and single FOV
                dataset = f'{dataset}_0'

            if feature_name is None or feature_name not in adata.var.columns:
                raise ValueError(f"feature_name {feature_name} not in adata.var. Please provide a valid feature name.")

            if label_key not in adata.obs.columns:
                raise ValueError(f"{label_key} not in adata.obs. Please double-check valid label name.")

            self.index = SCFind()
            self.index.buildCellTypeIndex(
                adata=adata,
                dataset_name=self.dataset,
                feature_name=feature_name,
                qb=2
            )
            self.genes = self.index.scfindGenes
        else:
            print('build_gene_index is False. Using adata.X for gene expression analysis.')
            # Store adata.X and gene names for direct vectorized DE analysis
            if if_lognorm:
                print('Log normalizing the expression data... If data is already log normalized, please set if_lognorm to False.')
                sc.pp.normalize_total(adata)
                sc.pp.log1p(adata)

            self.adata = adata
            self.genes = adata.var[feature_name].tolist()
            
    def get_genes(self):
        return self.genes
    
    def get_labels(self):
        return self.labels
    
    def get_dataset(self):
        return self.dataset


    def find_fp_knn(self,
                    ct: str,
                    k: int = 30,
                    min_support: float = 0.5,
                    max_dist: float = 20,
                    ) -> pd.DataFrame:
        """
        Find frequent patterns within the KNNs of certain cell type.

        Parameters
        ----------
        ct : str
            Cell type name.
        k : int, default=30
            Number of nearest neighbors.
        min_support : float, default=0.5
            Threshold of frequency to consider a pattern as a frequent pattern.
        max_dist : float, default=20
            Maximum distance for considering a cell as a neighbor.

        Returns
        -------
        pd.DataFrame
            Frequent patterns in the neighborhood of certain cell type.
        """
        if ct not in self.labels.unique():
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        cinds = [id for id, l in enumerate(self.labels) if l == ct]
        ct_pos = self.spatial_pos[cinds]

        fp, _, _ = spatial_utils.build_fptree_knn(
            kd_tree=self.kd_tree,
            labels=self.labels,
            cell_pos=ct_pos,
            k=k,
            min_support=min_support,
            max_dist=max_dist,
            if_max=True
        )

        return fp

    def find_fp_dist(self,
                     ct: str,
                     max_dist: float = 20,
                     min_size: int = 0,
                     min_support: float = 0.5,
                     ):
        """
        Find frequent patterns within the radius of certain cell type.

        Parameters
        ----------
        ct : str
            Cell type name.
        max_dist : float, default=20
            Maximum distance for considering a cell as a neighbor.
        min_size : int, default=0
            Minimum neighborhood size for each point to consider.
        min_support : float, default=0.5
            Threshold of frequency to consider a pattern as a frequent pattern.

        Returns
        -------
        pd.DataFrame
            Frequent patterns in the neighborhood of certain cell type.
        """
        if ct not in self.labels.unique():
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        cinds = [id for id, l in enumerate(self.labels) if l == ct]
        ct_pos = self.spatial_pos[cinds]

        fp, _, _ = spatial_utils.build_fptree_dist(
            kd_tree=self.kd_tree,
            labels=self.labels,
            cell_pos=ct_pos,
            max_dist=max_dist,
            min_size=min_size,
            min_support=min_support,
            cinds=cinds
        )

        return fp

    def motif_enrichment_knn(self,
                             ct: str,
                             motifs: Union[str, List[str], List[List[str]]] = None,
                             k: int = 30,
                             min_support: float = 0.5,
                             max_dist: float = 20,
                             return_cellID: bool = False
                             ) -> pd.DataFrame:
        """
        Perform motif enrichment analysis using k-nearest neighbors (KNN).

        Parameters
        ----------
        ct : str
            The cell type of the center cell.
        motifs : str or List[str] or List[List[str]], optional
            Specified motifs to be tested.
            If motifs=None, find the frequent patterns as motifs within the neighborhood of center cell type.
        k : int, default=30
            Number of nearest neighbors to consider.
        min_support : float, default=0.5
            Threshold of frequency to consider a pattern as a frequent pattern.
        max_dist : float, default=20
            Maximum distance for neighbors.
        return_cellID : bool, default=False
            Indicate whether return cell IDs for each frequent pattern within the neighborhood of grid points.
            By defaults do not return cell ID.

        Returns
        -------
        pd.DataFrame
            DataFrame with motif enrichment results. Columns include:
                - center: center cell type name
                - motifs: list of cell types in the motif
                - n_center_motif: number of center cells with motif in neighborhood
                - n_center: total number of center cells
                - n_motif: total number of cells with motif in neighborhood (across all cell types)
                - expectation: expected number under hypergeometric distribution
                - p-values: p-value from hypergeometric test
                - if_significant: whether the enrichment is significant (p < 0.05)
                - adj-pval: FDR-corrected p-values (only when multiple motifs tested)
                - neighbor_id: array of unique neighbor cell indices with motif types (only if return_cellID=True)
                - center_id: array of center cell indices with motif in neighborhood (only if return_cellID=True)
        """
        if ct not in self.labels.unique():
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        dists, idxs = self.kd_tree.query(self.spatial_pos,
                                         k=k + 1, workers=-1
                                         )  # use k+1 to find the knn except for the points themselves
        cinds = [i for i, l in enumerate(self.labels) if l == ct]

        out = []
        if motifs is None:
            fp = self.find_fp_knn(
                ct=ct, k=k,
                min_support=min_support,
                max_dist=max_dist,
                )
            motifs = fp['itemsets']
        else:
            if isinstance(motifs, str):
                motifs = [[motifs]]
            elif isinstance(motifs, list) and all(isinstance(m, str) for m in motifs):
                motifs = [motifs]
            # else: List[List[str]], keep as is

            labels_unique = self.labels.unique()
            filtered_motifs = []
            for motif in motifs:
                motif_exc = [m for m in motif if m not in labels_unique]
                if len(motif_exc) > 0:
                    print(f"Found no {motif_exc} in {self.label_key}. Ignoring them.")
                valid_motif = [m for m in motif if m in labels_unique]
                if len(valid_motif) > 0:
                    filtered_motifs.append(valid_motif)
            motifs = filtered_motifs

        if len(motifs) == 0:
            # Return empty DataFrame with same structure
            empty_df = pd.DataFrame(columns=['center', 'motifs', 'n_center_motif', 'n_center',
                                             'n_motif', 'expectation', 'p-values', 'if_significant'])
            return empty_df

        label_encoder = LabelEncoder()
        int_labels = label_encoder.fit_transform(self.labels)
        int_ct = label_encoder.transform(np.array(ct, dtype=object, ndmin=1))

        num_cells = idxs.shape[0]
        num_types = len(label_encoder.classes_)

        valid_neighbors = dists[:, 1:] <= max_dist
        filtered_idxs = np.where(valid_neighbors, idxs[:, 1:], -1)
        flat_neighbors = filtered_idxs.flatten()
        valid_neighbors_flat = valid_neighbors.flatten()

        neighbor_labels = np.where(valid_neighbors_flat, int_labels[flat_neighbors], -1)
        valid_mask = neighbor_labels != -1

        neighbor_matrix = np.zeros((num_cells * k, num_types), dtype=int)
        neighbor_matrix[np.arange(len(neighbor_labels))[valid_mask], neighbor_labels[valid_mask]] = 1

        neighbor_counts = neighbor_matrix.reshape(num_cells, k, num_types).sum(axis=1)

        mask = int_labels == int_ct

        for motif in motifs:
            motif = list(motif) if not isinstance(motif, list) else motif
            sort_motif = sorted(motif)

            int_motifs = label_encoder.transform(np.array(motif))

            n_motif_ct = np.sum(np.all(neighbor_counts[mask][:, int_motifs] > 0, axis=1))
            n_motif_labels = np.sum(np.all(neighbor_counts[:, int_motifs] > 0, axis=1))

            n_ct = len(cinds)
            if ct in motif:
                n_ct = round(n_ct / motif.count(ct))

            hyge = hypergeom(M=len(self.labels), n=n_ct, N=n_motif_labels)
            # M is number of total, N is number of drawn without replacement, n is number of success in total
            motif_out = {'center': ct, 'motifs': sort_motif, 'n_center_motif': n_motif_ct,
                         'n_center': n_ct, 'n_motif': n_motif_labels, 'expectation': hyge.mean(),
                         'p-values': hyge.sf(n_motif_ct)}

            if return_cellID:
                inds = np.where(np.all(neighbor_counts[mask][:, int_motifs] > 0, axis=1))[0]
                cind_with_motif = np.array(cinds)[inds]  # Centers with motif in neighborhood

                if len(cind_with_motif) == 0:
                    motif_out['neighbor_id'] = []
                    motif_out['center_id'] = []
                    continue

                motif_mask = np.isin(self.labels, motif)  # Mask for motif cell types

                # Use the idxs array which contains the original KNN indices
                # But filter by valid_neighbors which has distance filtering
                valid_idxs_of_centers = [
                    idxs[c, 1:][valid_neighbors[c, :]]  # Get valid neighbors by distance for each center
                    for c in cind_with_motif
                ]

                # Flatten and filter for motif types
                valid_neighbors_flat = np.concatenate(valid_idxs_of_centers)
                valid_motif_neighbors = valid_neighbors_flat[motif_mask[valid_neighbors_flat]]

                # Store unique IDs
                motif_out['neighbor_id'] = np.unique(valid_motif_neighbors)
                motif_out['center_id'] = cind_with_motif

            out.append(motif_out)

        out_pd = pd.DataFrame(out)
        
        if len(out_pd) == 0:
            out_pd = pd.DataFrame(columns=['center', 'motifs', 'n_center_motif', 'n_center',
                                           'n_motif', 'expectation', 'p-values', 'if_significant']) 
            return out_pd

        if len(out_pd) == 1:
            out_pd['if_significant'] = True if out_pd['p-values'][0] < 0.05 else False
            return out_pd
        else:
            p_values = out_pd['p-values'].tolist()
            if_rejected, corrected_p_values = mt.fdrcorrection(p_values,
                                                               alpha=0.05,
                                                               method='poscorr')
            out_pd['adj-pval'] = corrected_p_values
            out_pd['if_significant'] = if_rejected
            out_pd = out_pd.sort_values(by='adj-pval', ignore_index=True)
            return out_pd

    def motif_enrichment_dist(self,
                              ct: str,
                              motifs: Union[str, List[str], List[List[str]]] = None,
                              max_dist: float = 20,
                              min_size: int = 0,
                              min_support: float = 0.5,
                              return_cellID: bool = False,
                              ) -> DataFrame:
        """
        Perform motif enrichment analysis within a specified radius-based neighborhood.

        Parameters
        ----------
        ct : str
            Cell type as the center cells.
        motifs : str or List[str] or List[List[str]], optional
            Specified motifs to be tested.
            If motifs=None, find the frequent patterns as motifs within the neighborhood of center cell type.
        max_dist : float, default=20
            Maximum distance for considering a cell as a neighbor.
        min_size : int, default=0
            Minimum neighborhood size for each point to consider.
        min_support : float, default=0.5
            Threshold of frequency to consider a pattern as a frequent pattern.
        return_cellID : bool, default=False
            Indicate whether return cell IDs for each motif within the neighborhood of central cell type.
            By defaults do not return cell ID.

        Returns
        -------
        pd.DataFrame
            DataFrame with motif enrichment results. Columns include:
                - center: center cell type name
                - motifs: list of cell types in the motif
                - n_center_motif: number of center cells with motif in neighborhood
                - n_center: total number of center cells
                - n_motif: total number of cells with motif in neighborhood (across all cell types)
                - expectation: expected number under hypergeometric distribution
                - p-values: p-value from hypergeometric test
                - if_significant: whether the enrichment is significant (p < 0.05)
                - adj-pval: FDR-corrected p-values (only when multiple motifs tested)
                - neighbor_id: array of unique neighbor cell indices with motif types (only if return_cellID=True)
                - center_id: array of center cell indices with motif in neighborhood (only if return_cellID=True)
        """
        if ct not in self.labels.unique():
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        out = []
        if motifs is None:
            fp = self.find_fp_dist(ct=ct,
                                   max_dist=max_dist, min_size=min_size,
                                   min_support=min_support)
            motifs = fp['itemsets']
        else:
            if isinstance(motifs, str):
                motifs = [[motifs]]
            elif isinstance(motifs, list) and all(isinstance(m, str) for m in motifs):
                motifs = [motifs]
            # else: List[List[str]], keep as is

            labels_unique = self.labels.unique()
            filtered_motifs = []
            for motif in motifs:
                motif_exc = [m for m in motif if m not in labels_unique]
                if len(motif_exc) > 0:
                    print(f"Found no {motif_exc} in {self.label_key}. Ignoring them.")
                valid_motif = [m for m in motif if m in labels_unique]
                if len(valid_motif) > 0:
                    filtered_motifs.append(valid_motif)
            motifs = filtered_motifs

        if len(motifs) == 0:
            # Return empty DataFrame with same structure
            empty_df = pd.DataFrame(columns=['center', 'motifs', 'n_center_motif', 'n_center',
                                             'n_motif', 'expectation', 'p-values', 'if_significant'])
            return empty_df

        label_encoder = LabelEncoder()
        int_labels = label_encoder.fit_transform(np.array(self.labels))
        int_ct = label_encoder.transform(np.array(ct, dtype=object, ndmin=1))

        cinds = np.where(self.labels == ct)[0]

        num_cells = len(self.spatial_pos)
        num_types = len(label_encoder.classes_)

        # Query neighbors for all cells once (instead of using grid filtering)
        idxs_all = self.kd_tree.query_ball_point(
            self.spatial_pos,
            r=max_dist,
            return_sorted=False,
            workers=-1,
        )
        idxs_all_filter = [np.array(ids)[np.array(ids) != i] for i, ids in enumerate(idxs_all)]

        # Pre-compute neighbor matrix for all cells
        flat_neighbors_all = np.concatenate(idxs_all_filter)
        row_indices_all = np.repeat(np.arange(num_cells), [len(neigh) for neigh in idxs_all_filter])
        neighbor_labels_all = int_labels[flat_neighbors_all]

        neighbor_matrix_all = np.zeros((num_cells, num_types), dtype=int)
        np.add.at(neighbor_matrix_all, (row_indices_all, neighbor_labels_all), 1)

        for motif in motifs:
            motif = list(motif) if not isinstance(motif, list) else motif
            sort_motif = sorted(motif)

            # using numpy
            int_motifs = label_encoder.transform(np.array(motif))

            # Check which cells have all motif types in their neighborhood
            has_motif_mask = np.all(neighbor_matrix_all[:, int_motifs] > 0, axis=1)

            # Filter for center cell type
            mask_ct = int_labels == int_ct
            n_motif_ct = np.sum(has_motif_mask[mask_ct])
            n_motif_labels = np.sum(has_motif_mask)

            n_ct = len(cinds)
            if ct in motif:
                n_ct = round(n_ct / motif.count(ct))

            hyge = hypergeom(M=len(self.labels), n=n_ct, N=n_motif_labels)
            motif_out = {'center': ct, 'motifs': sort_motif, 'n_center_motif': n_motif_ct,
                         'n_center': n_ct, 'n_motif': n_motif_labels, 'expectation': hyge.mean(),
                         'p-values': hyge.sf(n_motif_ct)}

            if return_cellID:
                # Get center cells with motif
                cind_with_motif = cinds[has_motif_mask[cinds]]
                if len(cind_with_motif) == 0:
                    motif_out['center_id'] = np.array([])
                    motif_out['neighbor_id'] = np.array([])
                    continue

                # Get motif neighbors for these center cells
                motif_mask = np.isin(np.array(self.labels), motif)
                all_neighbors = np.concatenate([idxs_all_filter[i] for i in cind_with_motif])
                valid_neighbors = all_neighbors[motif_mask[all_neighbors]]
                id_motif_celltype = set(valid_neighbors)
                motif_out['neighbor_id'] = np.array(list(id_motif_celltype))
                motif_out['center_id'] = np.array(cind_with_motif)

            out.append(motif_out)

        out_pd = pd.DataFrame(out)
        
        if len(out_pd) == 0:
            out_pd = pd.DataFrame(columns=['center', 'motifs', 'n_center_motif', 'n_center',
                                           'n_motif', 'expectation', 'p-values', 'if_significant']) 
            return out_pd

        if len(out_pd) == 1:
            out_pd['if_significant'] = True if out_pd['p-values'][0] < 0.05 else False
            return out_pd
        else:
            p_values = out_pd['p-values'].tolist()
            if_rejected, corrected_p_values = mt.fdrcorrection(p_values,
                                                               alpha=0.05,
                                                               method='poscorr')
            out_pd['adj-pval'] = corrected_p_values
            out_pd['if_significant'] = if_rejected
            out_pd = out_pd.sort_values(by='adj-pval', ignore_index=True)
            return out_pd


    def find_patterns_grid(self,
                           max_dist: float = 20,
                           min_size: int = 0,
                           min_support: float = 0.5,
                           if_display: bool = True,
                           figsize: tuple = (10, 5),
                           return_cellID: bool = False,
                           return_grid: bool = False,
                           ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
        """
        Create a grid and use it to find surrounding patterns in spatial data.

        Parameters
        ----------
        max_dist : float, default=20
            Maximum distance to consider a cell as a neighbor. Also used as grid spacing.
        min_size : int, default=0
            Minimum neighborhood size for each grid point to consider.
        min_support : float, default=0.5
            Threshold of frequency to consider a pattern as a frequent pattern.
        if_display : bool, default=True
            Display the grid points with nearby frequent patterns if if_display=True.
        figsize : tuple, default=(10, 5)
            Tuple of figure size for the display plot.
        return_cellID : bool, default=False
            Indicate whether return cell IDs for each frequent pattern within the neighborhood of grid points.
            By default do not return cell ID.
        return_grid : bool, default=False
            Indicate whether return the grid points. By default, do not return grid points.
            If True, will return a tuple (fp_df, grid).

        Returns
        -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]
            If return_grid=False (default):
                pd.DataFrame with frequent pattern results. Columns include:
                    - itemsets: frozenset of cell types in the frequent pattern
                    - support: frequency of the pattern (proportion of grid points with this pattern)
                    - neighbor_id: set of cell indices belonging to the pattern (only if return_cellID=True)

            If return_grid=True:
                Tuple containing:
                    - pd.DataFrame: frequent pattern results as described above
                    - np.ndarray: grid points coordinates of shape (n_grid_points, 2)
        """

        xmax, ymax = np.max(self.spatial_pos, axis=0)
        xmin, ymin = np.min(self.spatial_pos, axis=0)
        x_grid = np.arange(xmin - max_dist, xmax + max_dist, max_dist)
        y_grid = np.arange(ymin - max_dist, ymax + max_dist, max_dist)
        grid = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)

        fp, trans_df, idxs = spatial_utils.build_fptree_dist(
            kd_tree=self.kd_tree,
            labels=self.labels,
            spatial_pos=self.spatial_pos,
            cell_pos=grid,
            max_dist=max_dist,
            min_size=min_size,
            min_support=min_support,
            if_max=True,
        )

        # For each frequent pattern/motif, locate the cell IDs in the neighborhood of the above grid points
        # as well as labelled with cell types in motif.
        # if dis_duplicates:
        #     normalized_columns = [col.split('_')[0] for col in trans_df.columns]
        #     trans_df.columns = normalized_columns
        #     sparse_trans_df = csr_matrix(trans_df, dtype=int)
        #     trans_df_aggregated = pd.DataFrame.sparse.from_spmatrix(sparse_trans_df, columns=normalized_columns)
        #     trans_df_aggregated = trans_df_aggregated.groupby(trans_df_aggregated.columns, axis=1).sum()
        id_neighbor_motifs = []
        if if_display or return_cellID:
            for motif in fp['itemsets']:
                motif = list(motif)
                fp_spots_index = set()
                # if dis_duplicates:
                #     ct_counts_in_motif = pd.Series(motif).value_counts().to_dict()
                #     required_counts = pd.Series(ct_counts_in_motif, index=trans_df_aggregated.columns).fillna(0)
                #     ids = trans_df_aggregated[trans_df_aggregated >= required_counts].dropna().index
                # else:
                #     ids = trans_df[trans_df[motif].all(axis=1)].index.to_list()
                ids = trans_df[trans_df[motif].all(axis=1)].index.to_list()
                if isinstance(idxs, list):
                    # ids = ids.index[ids == True].to_list()
                    fp_spots_index.update([i for id in ids for i in idxs[id] if self.labels[i] in motif])
                else:
                    ids = idxs[ids]
                    fp_spots_index.update([i for id in ids for i in id if self.labels[i] in motif])
                id_neighbor_motifs.append(fp_spots_index)
        if return_cellID:
            fp['neighbor_id'] = id_neighbor_motifs

        if if_display:
            fp_cts = sorted(set(t for items in fp['itemsets'] for t in list(items)))
            n_colors = len(fp_cts)
            colors = sns.color_palette('hsv', n_colors)
            color_map = {ct: col for ct, col in zip(fp_cts, colors)}

            fp_spots_index = set()
            for cell_id in id_neighbor_motifs:
                fp_spots_index.update(cell_id)

            fp_spot_pos = self.spatial_pos[list(fp_spots_index), :]
            fp_spot_label = self.labels[list(fp_spots_index)]
            fig, ax = plt.subplots(figsize=figsize)
            # Plotting the grid lines
            for x in x_grid:
                ax.axvline(x, color='lightgray', linestyle='--', lw=0.5)

            for y in y_grid:
                ax.axhline(y, color='lightgray', linestyle='--', lw=0.5)

            for ct in fp_cts:
                ct_ind = fp_spot_label == ct
                ax.scatter(fp_spot_pos[ct_ind, 0], fp_spot_pos[ct_ind, 1],
                           label=ct, color=color_map[ct], s=1)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)
            plt.xlabel('Spatial X')
            plt.ylabel('Spatial Y')
            plt.title('Spatial distribution of frequent patterns')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            # plt.tight_layout(rect=[0, 0, 1.1, 1])
            plt.show()

        if return_grid:
            return fp.sort_values(by='support', ignore_index=True, ascending=False), grid
        else:
            return fp.sort_values(by='support', ignore_index=True, ascending=False)

    def find_patterns_rand(self,
                           max_dist: float = 20,
                           n_points: int = 1000,
                           min_support: float = 0.5,
                           min_size: int = 0,
                           if_display: bool = True,
                           figsize: tuple = (10, 5),
                           return_cellID: bool = False,
                           seed: int = 2023) -> pd.DataFrame:
        """
        Randomly generate points and use them to find surrounding patterns in spatial data.

        Parameters
        ----------
        max_dist : float, default=20
            Maximum distance to consider a cell as a neighbor.
        n_points : int, default=1000
            Number of random points to generate.
        min_support : float, default=0.5
            Threshold of frequency to consider a pattern as a frequent pattern.
        min_size : int, default=0
            Minimum neighborhood size for each random point to consider.
        if_display : bool, default=True
            Display the random points with nearby frequent patterns if if_display=True.
        figsize : tuple, default=(10, 5)
            Tuple of figure size for the display plot.
        return_cellID : bool, default=False
            Indicate whether return cell IDs for each frequent pattern within the neighborhood of random points.
            By default do not return cell ID.
        seed : int, default=2023
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            DataFrame with frequent pattern results. Columns include:
                - itemsets: frozenset of cell types in the frequent pattern
                - support: frequency of the pattern (proportion of random points with this pattern)
                - neighbor_id: set of cell indices belonging to the pattern (only if return_cellID=True)
        """

        xmax, ymax = np.max(self.spatial_pos, axis=0)
        xmin, ymin = np.min(self.spatial_pos, axis=0)
        np.random.seed(seed)
        pos = np.column_stack((np.random.rand(n_points) * (xmax - xmin) + xmin,
                               np.random.rand(n_points) * (ymax - ymin) + ymin))

        fp, trans_df, idxs = spatial_utils.build_fptree_dist(
            kd_tree=self.kd_tree,
            labels=self.labels,
            spatial_pos=self.spatial_pos,
            cell_pos=pos,
            max_dist=max_dist,
            min_size=min_size,
            min_support=min_support,
            if_max=True,
        )
        # if dis_duplicates:
        #     normalized_columns = [col.split('_')[0] for col in trans_df.columns]
        #     trans_df.columns = normalized_columns
        #     sparse_trans_df = csr_matrix(trans_df, dtype=int)
        #     trans_df_aggregated = pd.DataFrame.sparse.from_spmatrix(sparse_trans_df, columns=normalized_columns)
        #     trans_df_aggregated = trans_df_aggregated.groupby(trans_df_aggregated.columns, axis=1).sum()

        id_neighbor_motifs = []
        if if_display or return_cellID:
            for motif in fp['itemsets']:
                motif = list(motif)
                fp_spots_index = set()
                # if dis_duplicates:
                #     ct_counts_in_motif = pd.Series(motif).value_counts().to_dict()
                #     required_counts = pd.Series(ct_counts_in_motif, index=trans_df_aggregated.columns).fillna(0)
                #     ids = trans_df_aggregated[trans_df_aggregated >= required_counts].dropna().index
                # else:
                #     ids = trans_df[trans_df[motif].all(axis=1)].index.to_list()
                ids = trans_df[trans_df[motif].all(axis=1)].index.to_list()
                if isinstance(idxs, list):
                    # ids = ids.index[ids == True].to_list()
                    fp_spots_index.update([i for id in ids for i in idxs[id] if self.labels[i] in motif])
                else:
                    ids = idxs[ids]
                    fp_spots_index.update([i for id in ids for i in id if self.labels[i] in motif])
                id_neighbor_motifs.append(fp_spots_index)
        if return_cellID:
            fp['neighbor_id'] = id_neighbor_motifs

        if if_display:
            fp_cts = sorted(set(t for items in fp['itemsets'] for t in list(items)))
            n_colors = len(fp_cts)
            colors = sns.color_palette('hsv', n_colors)
            color_map = {ct: col for ct, col in zip(fp_cts, colors)}

            fp_spots_index = set()
            for cell_id in id_neighbor_motifs:
                fp_spots_index.update(cell_id)

            fp_spot_pos = self.spatial_pos[list(fp_spots_index), :]
            fp_spot_label = self.labels[list(fp_spots_index)]
            fig, ax = plt.subplots(figsize=figsize)
            for ct in fp_cts:
                ct_ind = fp_spot_label == ct
                ax.scatter(fp_spot_pos[ct_ind, 0], fp_spot_pos[ct_ind, 1],
                           label=ct, color=color_map[ct], s=1)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)
            plt.xlabel('Spatial X')
            plt.ylabel('Spatial Y')
            plt.title('Spatial distribution of frequent patterns')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout(rect=[0, 0, 1.1, 1])
            plt.show()

        return fp.sort_values(by='support', ignore_index=True, ascending=False)

    def de_genes(self,
                 ind_group1: List[int],
                 ind_group2: List[int],
                 genes: Optional[Union[str, List[str]]] = None,
                 min_fraction: float = 0.05,
                 method: Literal['fisher', 't-test', 'wilcoxon'] = 'fisher',
                 alpha: Optional[float] = None,
                 ) -> pd.DataFrame:
        """
        Identify differential genes between two groups of cells.

        Parameters
        ----------
        ind_group1 : List[int]
            List of indices of cells in group 1.
        ind_group2 : List[int]
            List of indices of cells in group 2.
        genes : str or List[str], optional
            List of gene names to query. If None, all genes will be used.
        min_fraction : float, default=0.05
            The minimum fraction of cells that express a gene for it to be considered differentially expressed.
        method : {'fisher', 't-test', 'wilcoxon'}, default='fisher'
            The method to use for DE analysis. If build_gene_index=True, only Fisher's exact test is supported.
        alpha : float, optional
            Significance threshold for adjusted p-values. If None, defaults to 0.1 when using Fisher's exact test and 0.05 otherwise.

        Returns
        -------
        pd.DataFrame containing the differentially expressed genes between the two groups.
        """
        if self.build_gene_index:
            # Use scfind index for DE analysis with Fisher's exact test.
            if genes is None:
                genes = self.genes
                
            if method != 'fisher':
                print(f"Warning: When build_gene_index=True, only Fisher's exact test is supported. Ignoring method='{method}'.")
            
            out = self.index.de_genes_with_indices(genes, ind_group1, ind_group2, min_fraction)
            out_df = pd.DataFrame(out)

            print(f"number of tested genes using scfind index: {len(out_df)}")

            # Calculate Fisher's exact test p-values in Python using scipy
            p_values = []
            for _, row in out_df.iterrows():
                table = [[int(row['a']), int(row['b'])], 
                        [int(row['c']), int(row['d'])]]
                _, p_value = fisher_exact(table, alternative='two-sided')
                p_values.append(p_value)
            
            out_df['p_value'] = p_values

            adjusted_pvals = multipletests(out_df['p_value'], method='fdr_bh')[1]
            out_df['adj-pval'] = adjusted_pvals

            if alpha is None:
                alpha = 0.1

            results_df = out_df[out_df['adj-pval'] < alpha]
            results_df.loc[:, 'de_in'] = np.where(
                (results_df['proportion_1'] >= results_df['proportion_2']),
                'group1',
                np.where(
                    (results_df['proportion_2'] > results_df['proportion_1']),
                    'group2',
                    None
                )
            )
            results_df = results_df[results_df['adj-pval'] < alpha].sort_values('p_value').reset_index(drop=True)
            results_df = results_df[['gene', 'proportion_1', 'proportion_2', 'abs_difference', 'p_value', 'adj-pval', 'de_in']]
        else:
            # Use adata.X directly for DE analysis
            if method == 'fisher':
                results_df = spatial_utils.de_genes_fisher(
                    self.adata, self.genes, ind_group1, ind_group2, genes, min_fraction, alpha
                )
                results_df = results_df[['gene', 'proportion_1', 'proportion_2', 'abs_difference', 'p_value', 'adj-pval', 'de_in']]
            elif method == 't-test' or method == 'wilcoxon':
                results_df = spatial_utils.de_genes_scanpy(
                    self.adata, self.genes, ind_group1, ind_group2, genes, min_fraction, method=method, alpha=alpha
                )
                results_df = results_df[['gene', 'proportion_1', 'proportion_2', 'abs_difference', 'log2fc', 'p_value', 'adj-pval', 'de_in']]
            else:
                raise ValueError(f"Invalid method: {method}. Choose from 'fisher', 't-test', or 'wilcoxon'.")

        return results_df

    def get_anchor_motif_cell_ids(self,
                        ct: str,
                        motif: Union[str, List[str]],
                        max_dist: Optional[float] = None,
                        k: Optional[int] = None,
                        min_size: int = 0,
                        ) -> dict:
        """
        Get cell grouping information for correlation analysis without computing correlations.

        Parameters
        ----------
        ct : str
            Cell type as the center cells.
        motif : str or List[str]
            Motif (names of cell types) to be analyzed.
        max_dist : float, optional
            Maximum distance for considering a cell as a neighbor. Use either max_dist or k.
        k : int, optional
            Number of nearest neighbors. Use either max_dist or k.
        min_size : int, default=0
            Minimum neighborhood size for each center cell (only used when max_dist is specified).

        Returns
        -------
        cell_groups : dict
            Dictionary containing cell pairing information:
                - 'center_neighbor_motif_pair': array of [center, neighbor] index pairs.
                - 'non-neighbor_motif_cells': distant motif cell indices.
                - 'non_motif_center_neighbor_pair': pairs for centers without nearby motif.
        """
        return spatial_gene_covarying.get_anchor_motif_cell_ids(
            sq_obj=self,
            ct=ct,
            motif=motif,
            max_dist=max_dist,
            k=k,
            min_size=min_size,
        )

    def compute_gene_gene_correlation(self,
                                      ct: str,
                                      motif: Union[str, List[str]],
                                      genes: Optional[Union[str, List[str]]] = None,
                                      max_dist: Optional[float] = None,
                                      k: Optional[int] = None,
                                      min_size: int = 0,
                                      min_nonzero: int = 10,
                                      alpha: Optional[float] = None
                                      ) -> Tuple[pd.DataFrame, dict]:
        """
        Compute gene-gene cross correlation between anchor and neighboring motif cells. Only considers inter-cell-type interactions. 
        After finding neighbors using the full motif, removes all cells of the center cell type from both neighbor and
        non-neighbor groups. For Pearson correlation, uses shifted correlation (subtract cell type mean) to enable
        comparison across different niches/motifs.

        This function calculates cross correlation between gene expression in motif cells that are
        neighbors of the center type, motif cells that are not neighbors, and neighboring cells
        without nearby motif. Center type cells are excluded from neighbor groups in all cases.

        Parameters
        ----------
        ct : str
            Cell type as the center cells.
        motif : str or List[str]
            Motif (names of cell types) to be analyzed. Include all cell types for neighbor finding.
        genes : str or List[str], optional
            List of genes to analyze. If None, all genes will be used.
        max_dist : float, optional
            Maximum distance for considering a cell as a neighbor. Use either max_dist or k.
        k : int, optional
            Number of nearest neighbors. Use either max_dist or k.
        min_size : int, default=0
            Minimum neighborhood size for each center cell (only used when max_dist is specified).
        min_nonzero : int, default=10
            Minimum number of non-zero expression values required for a gene to be included.
        alpha : float, optional
            Significance threshold.

        Returns
        -------
        results_df : pd.DataFrame
            DataFrame with correlation results. Columns: gene_center, gene_motif, corr_neighbor,
            corr_non_neighbor, p_value_test1, delta_corr_test1, corr_center_no_motif,
            p_value_test2, delta_corr_test2, q_value_test1, q_value_test2, reject_test1_fdr,
            reject_test2_fdr, combined_score, abs_combined_score, if_significant.
        cell_groups : dict
            Cell pairing info. Keys: 'center_neighbor_motif_pair' (array of [center, neighbor]
            index pairs), 'non-neighbor_motif_cells' (distant motif cell indices),
            'non_motif_center_neighbor_pair' (pairs for centers without motif).
        """
        if not self.build_gene_index:
            print('Computing covarying genes using expression data...')
            results_df, ids = spatial_gene_covarying.compute_gene_gene_correlation_adata(
                sq_obj=self,
                ct=ct,
                motif=motif,
                genes=genes,
                max_dist=max_dist,
                k=k,
                min_size=min_size,
                min_nonzero=min_nonzero,
                alpha=alpha,
            )
        else:
            print('Computing covarying genes using binary data...')
            results_df, ids = spatial_gene_covarying.compute_gene_gene_correlation_binary(
                sq_obj=self,
                ct=ct,
                motif=motif,
                genes=genes,
                max_dist=max_dist,
                k=k,
                min_size=min_size,
                min_nonzero=min_nonzero,
                alpha=alpha
            )

        return results_df, ids


    def compute_gene_gene_correlation_by_type(self,
                                             ct: str,
                                             motif: Union[str, List[str]],
                                             genes: Optional[Union[str, List[str]]] = None,
                                             max_dist: Optional[float] = None,
                                             k: Optional[int] = None,
                                             min_size: int = 0,
                                             min_nonzero: int = 10,
                                             alpha: Optional[float] = None,
                                             ) -> pd.DataFrame:
        """
        Compute gene-gene cross correlation separately for each cell type in the motif.
        For each non-center cell type in the motif, compute:
        - Correlation 1: Center cells with motif vs neighboring motif cells of THIS TYPE
        - Correlation 2: Center cells with motif vs distant motif cells of THIS TYPE
        - Correlation 3: Center cells without motif vs neighbors (same for all types)

        Only analyzes motifs with >= 2 cell types besides the center type.

        Parameters
        ----------
        ct : str
            Cell type as the center cells.
        motif : str or List[str]
            Motif (names of cell types) to be analyzed. Include all cell types for neighbor finding.
        genes : str or List[str], optional
            List of genes to analyze. If None, all genes will be used.
        max_dist : float, optional
            Maximum distance for considering a cell as a neighbor. Use either max_dist or k.
        k : int, optional
            Number of nearest neighbors. Use either max_dist or k.
        min_size : int, default=0
            Minimum neighborhood size for each center cell (only used when max_dist is specified).
        min_nonzero : int, default=10
            Minimum number of non-zero expression values required for a gene to be included.
        alpha : float, optional
            Significance threshold.

        Returns
        -------
        pd.DataFrame
            DataFrame with correlation results per cell type and gene pair. Columns: cell_type,
            gene_center, gene_motif, corr_neighbor, corr_non_neighbor, corr_center_no_motif,
            p_value_test1, p_value_test2, q_value_test1, q_value_test2, delta_corr_test1,
            delta_corr_test2, reject_test1_fdr, reject_test2_fdr, combined_score,
            abs_combined_score, if_significant.
        """
        if not self.build_gene_index:
            print('Computing covarying genes using expression data...')
            return spatial_gene_covarying.compute_gene_gene_correlation_by_type_adata(
                sq_obj=self,
                ct=ct,
                motif=motif,
                genes=genes,
                max_dist=max_dist,
                k=k,
                min_size=min_size,
                min_nonzero=min_nonzero,
                alpha=alpha,
            )
        else: 
            print('Computing covarying genes using binary data...')
            return spatial_gene_covarying.compute_gene_gene_correlation_by_type_binary(
                sq_obj=self,
                ct=ct,
                motif=motif,
                genes=genes,
                max_dist=max_dist,
                k=k,
                min_size=min_size,
                min_nonzero=min_nonzero,
                alpha=alpha,
            )
        
    

    

    @staticmethod
    def test_score_difference(
        result_A: pd.DataFrame,
        result_B: pd.DataFrame,
        score_col: str = 'combined_score',
        significance_col: str = 'if_significant',
        gene_center_col: str = 'gene_center',
        gene_motif_col: str = 'gene_motif',
        percentile_threshold: float = 95.0,
        background: Literal['Overlapping', 'Significant'] = 'Significant'
        ) -> pd.DataFrame:
        """
        Identify gene pairs with large score differences between two covariation pattern results.

        This function compares covariation scores between two groups (e.g., disease vs control,
        treatment vs baseline, covarying gene pairs by distinct motif types) and identifies gene 
        pairs with the largest score differences using percentile-based ranking. Gene pairs in the 
        top percentile_threshold% and bottom (100 - percentile_threshold)% of score differences are 
        flagged as emperical significant ones.

        Parameters
        ----------
        result_A : pd.DataFrame
            Results from compute_gene_gene_correlation or compute_gene_gene_correlation_by_type
            for condition A. Must contain columns: gene_center, gene_motif, combined_score, if_significant.
        result_B : pd.DataFrame
            Results from compute_gene_gene_correlation or compute_gene_gene_correlation_by_type
            for condition B. Must contain the same columns as result_A.
        score_col : str, default='combined_score'
            Name of the column containing covariation scores to compare.
        significance_col : str, default='if_significant'
            Name of the column indicating whether a gene pair is significant.
        gene_center_col : str, default='gene_center'
            Name of the column containing center gene names.
        gene_motif_col : str, default='gene_motif'
            Name of the column containing motif gene names.
        percentile_threshold : float, default=95.0
            Percentile threshold for identifying outliers. Gene pairs with score_diff in the
            top percentile_threshold% (e.g., >95th percentile) or bottom (100 - percentile_threshold)%
            (e.g., <5th percentile) are flagged as outliers.
        background : Literal['Overlapping', 'Significant'], default='Significant'
            Defines the background gene pairs for comparison:
            - 'Significant': Only gene pairs significant in at least one condition are considered.
            - 'Overlapping': All overlapping gene pairs between both conditions are considered.

        Returns
        -------
        pd.DataFrame
            DataFrame with gene pair comparison results. Columns include:
                - gene_center: center gene name
                - gene_motif: motif gene name
                - score_A: covariation score in condition A
                - score_B: covariation score in condition B
                - score_diff: score difference (score_A - score_B)
                - percentile: percentile rank of score_diff in the distribution
                - is_outlier: True if percentile > percentile_threshold or < (100 - percentile_threshold)
                - significant_in_A: whether the gene pair is significant in condition A
                - significant_in_B: whether the gene pair is significant in condition B
                - outlier_direction: 'higher_in_A' (top percentile), 'higher_in_B' (bottom percentile),
                  or 'not_outlier'
        """

        return spatial_gene_covarying.test_score_difference(result_A, result_B, score_col, significance_col, gene_center_col, gene_motif_col, percentile_threshold, background)
    
    def plot_fov(self,
                 min_cells_label: int = 50,
                 title: str = 'Spatial distribution of cell types',
                 figsize: tuple = (10, 5),
                 save_path: Optional[str] = None,
                 ) -> None:
        """
        Plot the cell type distribution of a single field of view (FOV).

        Parameters
        ----------
        min_cells_label : int, default=50
            Minimum number of cells required for a cell type to be displayed in the plot.
            Cell types with fewer cells will be excluded from the legend.
        title : str, default='Spatial distribution of cell types'
            Title of the figure.
        figsize : tuple, default=(10, 5)
            Figure size as (width, height) tuple.
        save_path : str, optional
            Path to save the figure. If None, the figure will not be saved.

        Returns
        -------
        None
            Displays a scatter plot showing spatial distribution of cell types,
            with each cell type colored differently. The legend shows cell types
            with at least min_cells_label cells.
        """
        return plotting.plot_fov(sq_obj=self, min_cells_label=min_cells_label, title=title, figsize=figsize, save_path=save_path)

    def plot_motif_grid(self,
                        motif: Union[str, List[str]],
                        figsize: tuple = (10, 5),
                        max_dist: float = 20,
                        save_path: Optional[str] = None
                        ):
        """
        Display the distribution of each motif around grid points.

        Parameters
        ----------
        motif : str or List[str]
            Motif (names of cell types) to be colored.
        figsize : tuple, default=(10, 5)
            Figure size.
        max_dist : float, default=20
            Spacing distance for building grid.
        save_path : str, optional
            Path to save the figure. If None, the figure will not be saved.

        Returns
        -------
        None
            Displays a figure.
        """
        return plotting.plot_motif_grid(sq_obj=self, motif=motif, figsize=figsize, max_dist=max_dist, save_path=save_path)

    def plot_motif_rand(self,
                        motif: Union[str, List[str]],
                        max_dist: float = 100,
                        n_points: int = 1000,
                        figsize: tuple = (10, 5),
                        seed: int = 2023,
                        save_path: Optional[str] = None
                        ):
        """
        Display the random sampled points with motif in radius-based neighborhood,
        and cell types of motif in the neighborhood of these random points.

        Parameters
        ----------
        motif : str or List[str]
            Motif (names of cell types) to be colored.
        max_dist : float, default=100
            Radius for neighborhood search.
        n_points : int, default=1000
            Number of random points to generate.
        figsize : tuple, default=(10, 5)
            Figure size.
        seed : int, default=2023
            Random seed for reproducibility.
        save_path : str, optional
            Path to save the figure. If None, the figure will not be saved.

        Returns
        -------
        None
            Displays a figure.
        """
        return plotting.plot_motif_rand(sq_obj=self, motif=motif, max_dist=max_dist, n_points=n_points, figsize=figsize, seed=seed, save_path=save_path)


    def plot_motif_celltype(self,
                            ct: str,
                            motif: Union[str, List[str]],
                            max_dist: float = 20,
                            figsize: tuple = (5, 5),
                            save_path: Optional[str] = None
                            ):
        """
        Display the distribution of interested motifs in the radius-based neighborhood of certain cell type.
        This function could be used to visualize significant motifs by `motif_enrichment_dist()` 
        or `motif_enrichment_knn()`, or customized motifs by users.


        Parameters
        ----------
        ct : str
            Cell type as the center cells.
        motif : str or List[str]
            Motif (names of cell types) to be colored.
        max_dist : float, default=20
            Spacing distance for building grid. Make sure using the same value as that in find_patterns_grid.
        figsize : tuple, default=(5, 5)
            Figure size.
        save_path : str, optional
            Path to save the figure. If None, the figure will not be saved.

        Returns
        -------
        None
            Displays a figure.
        """
        return plotting.plot_motif_celltype(sq_obj=self, ct=ct, motif=motif, max_dist=max_dist, figsize=figsize, save_path=save_path)

    def plot_all_center_motif(
            self,
            ct: str,
            ids: dict,
            figsize: tuple = (6, 6),
            save_path: Optional[str] = None,
            ):
        """
        Plot the cell type distribution of single fov.

        Parameters
        ----------
        ct : str
            Center cell type.
        ids : dict
            Dictionary containing cell pairing information for correlations:
                - 'center_neighbor_motif_pair': array of shape (n_pairs, 2) containing
                  center-neighbor pairs for Correlation 1 (center with motif vs neighboring motif).
                  Each row is [center_cell_idx, neighbor_cell_idx].
                - 'non-neighbor_motif_cells': array of cell indices for distant motif cells
                  used in Correlation 2 (center with motif vs distant motif).
                  Correlation 2 uses all combinations of center cells (from corr1) × these cells.
                - 'non_motif_center_neighbor_pair': array of shape (n_pairs, 2) containing
                  center-neighbor pairs for Correlation 3 (center without motif vs neighbors).
                  Each row is [center_cell_idx, neighbor_cell_idx]. Empty if insufficient pairs.
        figsize : tuple, default=(6, 6)
            Figure size.
        save_path : str, optional
            Path to save the figure. If None, the figure will not be saved.

        Returns
        -------
        None
            Displays a figure.
        """
        return plotting.plot_all_center_motif(sq_obj=self, ct=ct, ids=ids, figsize=figsize, save_path=save_path)

    def plot_fp_heatmap(self,
                        fp_df: pd.DataFrame,
                        figsize: tuple = (7, 5),
                        save_path: Optional[str] = None,
                        title: Optional[str] = None,
                        cmap: str = 'GnBu'):
        """
        Plot a heatmap showing the distribution of cell types in frequent patterns.

        Parameters
        ----------
        fp_df : pd.DataFrame
            DataFrame containing frequent pattern results with two columns:
                - support: frequency of the pattern (proportion of points with this pattern)
                - itemsets: frozenset of cell types in the frequent pattern

            This is typically the output from find_fp_knn or find_fp_dist methods.
        figsize : tuple, default=(7, 5)
            Figure size.
        save_path : str, optional
            Path to save the figure. If None, the figure will not be saved.
        title : str, optional
            Figure title. If None, will use a default title.
        cmap : str, default='GnBu'
            Colormap for the heatmap.

        Returns
        -------
        None
            Displays a heatmap showing the cell type distribution across frequent patterns.
            Rows represent cell types, columns represent pattern groups (sorted by support),
            and cell values show the support frequency.
        """
        return plotting.plot_fp_heatmap(fp_df=fp_df, figsize=figsize,
                                        save_path=save_path, title=title, cmap=cmap)

    def plot_motif_enrichment_heatmap(self,
                                       enrich_df: pd.DataFrame,
                                       figsize: tuple = (7, 5),
                                       save_path: Optional[str] = None,
                                       title: Optional[str] = None,
                                       cmap: str = 'GnBu'):
        """
        Plot a heatmap showing the distribution of cell types in enriched motifs.

        Parameters
        ----------
        enrich_df : pd.DataFrame
            Output DataFrame from motif_enrichment_dist or motif_enrichment_knn.
        figsize : tuple, default=(7, 5)
            Figure size.
        save_path : str, optional
            Path to save the figure. If None, the figure will not be saved.
        title : str, optional
            Figure title. If None, will use a default title based on center cell type.
        cmap : str, default='GnBu'
            Colormap for the heatmap.

        Returns
        -------
        None
            Displays a heatmap of motif cell type distribution.
        """
        return plotting.plot_motif_enrichment_heatmap(enrich_df=enrich_df, figsize=figsize,
                                                       save_path=save_path, title=title, cmap=cmap)
        
    def plot_gene_pair_heatmap(self,
                               gene_pair_df: pd.DataFrame,
                               figsize: tuple = (7, 5),
                               save_path: Optional[str] = None,
                               title: Optional[str] = None,
                               cmap: str = 'GnBu'):
        """
        Plot a heatmap showing the cross-varying gene pairs and return cluster assignments.

        Parameters
        ----------
        gene_pair_df : pd.DataFrame
            Output DataFrame from compute_gene_gene_correlation or compute_gene_gene_correlation_by_type.
        figsize : tuple, default=(7, 5)
            Figure size.
        save_path : str, optional
            Path to save the figure. If None, the figure will not be saved.
        title : str, optional
            Figure title. If None, will use a default title based on center cell type.
        cmap : str, default='GnBu'
            Colormap for the heatmap.

        Returns
        -------
        pd.DataFrame
            Displays a biclustered heatmap of gene pairs and returns a DataFrame with
            cluster assignments. Columns: gene_center, gene_motif, combined_score,
            cluster_type, cluster_row, cluster_col. If input has 'cell_type' column,
            it will also be included.
        """
        return plotting.plot_gene_pair_heatmap(gene_pair_df=gene_pair_df, figsize=figsize,
                                                       save_path=save_path)

    def plot_gene_pair_spatial(self,
                               gene_pairs: List[tuple],
                               gene_pair_df: pd.DataFrame,
                               ids: dict,
                               ct: str,
                               motif: List[str],
                               motif_ct: Optional[str] = None,
                               figsize: tuple = (20, 5),
                               vmin: Optional[float] = None,
                               vmax: Optional[float] = None,
                               cmap: str = "RdYlBu_r",
                               save_path: Optional[str] = None,
                               ):
        """
        Plot spatial expression distribution of gene pairs showing shifted expression.

        For each gene pair, generates 4 panels showing:
        1. Center gene expression in center cells with motif
        2. Motif gene expression in neighboring motif cells
        3. Center gene expression in center cells without motif
        4. Motif gene expression in non-motif neighbors

        Parameters
        ----------
        gene_pairs : List[tuple]
            List of gene pairs to plot, e.g., [('gene1', 'gene2'), ('gene3', 'gene4')]. First gene
            in the pair is the center gene, second is the motif gene.
        gene_pair_df : pd.DataFrame
            DataFrame containing gene pair correlation results with columns:
            'gene_center', 'gene_motif', 'combined_score'.
            This is typically the output from compute_gene_gene_correlation.
        ids : dict
            Dictionary containing cell pairing information from compute_gene_gene_correlation:
            - 'center_neighbor_motif_pair': array of [center_idx, neighbor_idx] pairs
            - 'non_motif_center_neighbor_pair': array of [center_idx, neighbor_idx] pairs
        ct : str
            Center cell type name.
        motif : List[str]
            List of cell types in the motif.
        motif_ct : str, optional
            Specific motif cell type to filter for plotting. If None, all motif cell types
            are used (suitable for compute_gene_gene_correlation results where motif types
            are pooled together).
        figsize : tuple, default=(20, 5)
            Figure size for each gene pair plot (width, height).
        vmin : float, optional
            Minimum value for color normalization. If None, automatically computed as
            ``-max(abs(shifted_expression))`` to ensure symmetric colormap.
        vmax : float, optional
            Maximum value for color normalization. If None, automatically computed as
            ``max(abs(shifted_expression))`` to ensure symmetric colormap.
        cmap : str, default='RdYlBu_r'
            Colormap name.
        save_path : str, optional
            Directory path to save figures. If None, figures are displayed but not saved.
            Each gene pair will be saved as '{save_path}/{gene1}_{gene2}_spatial.pdf'.

        Returns
        ------
        None
            Displays and optionally saves spatial expression plots for each gene pair.

        Example
        -------
        >>> gene_pair_df, ids = sq.compute_gene_gene_correlation(
        ...     ct='Cardiomyocyte',
        ...     motif=['Fibroblast', 'Endothelial'],
        ...     max_dist=20
        ... )
        >>> sq.plot_gene_pair_spatial(
        ...     gene_pairs=[('Ttn', 'Myh6'), ('Actn2', 'Tnni3')],
        ...     gene_pair_df=gene_pair_df,
        ...     ids=ids,
        ...     ct='Cardiomyocyte',
        ...     motif=['Fibroblast', 'Endothelial'],
        ...     motif_ct='Fibroblast',
        ...     save_path='./figures/'
        ... )
        """
        return plotting.plot_gene_pair_spatial(
            sq_obj=self,
            gene_pairs=gene_pairs,
            gene_pair_df=gene_pair_df,
            ids=ids,
            ct=ct,
            motif=motif,
            motif_ct=motif_ct,
            figsize=figsize,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            save_path=save_path,
        )
        
def interactive_motif(
    sp,
    zarr_path: str,
    spatialSpotRadius: float = 1,
):
    """
    Create an interactive Vitessce widget to explore spatial motifs.

    Builds a minimal AnnData containing cell type labels, spatial coordinates,
    and (optionally) gene expression from the SpatialQuery object, writes it
    to a Zarr store, and launches a Vitessce widget with the SpatialQueryPlugin.
    The widget provides interactive panels for spatial visualization, cell type
    selection, motif enrichment analysis, and gene expression exploration.

    Parameters
    ----------
    sp : spatial_query
        An initialized SpatialQuery instance. If ``build_gene_index=False``
        and ``sp.adata`` is not None, gene expression data will be included
        in the widget for feature-level exploration.
    zarr_path : str
        Path where the Zarr store will be written. If the path already exists,
        its contents will be overwritten with a warning.
    spatialSpotRadius : float, default=1
        Radius for spatial spots in the Vitessce visualization. Adjust based
        on data density and spatial coordinate scale.

    Returns
    -------
    ipywidgets.Widget
        A Vitessce widget that can be displayed in a Jupyter notebook.
        The widget contains the following views:

        - **Spatial view**: scatter plot of cells colored by cell type
        - **Layer controller**: toggle visibility and appearance of layers
        - **Cell sets**: browse and select cell types
        - **SpatialQuery**: interactive motif enrichment query panel
        - **SpatialQuery Heatmap**: heatmap of motif enrichment results
        - **Feature list**: gene selection panel (only when expression data is available)

    Notes
    -----
    This function requires the ``vitessce`` Python package (with the
    ``SpatialQueryPlugin``) to be installed. Install it via::

        pip install vitessce[all]

    The generated widget is designed for use in Jupyter notebooks
    (JupyterLab or Jupyter Notebook).

    Examples
    --------
    >>> from SpatialQuery.spatial_query import spatial_query, interactive_motif
    >>> sq = spatial_query(adata, spatial_key='X_spatial', label_key='cell_type')
    >>> widget = interactive_motif(sq, zarr_path='./my_data.zarr')
    >>> widget
    """
    import os
    import warnings
    import anndata
    import scipy.sparse as ssp
    from vitessce import (
        VitessceConfig,
        AnnDataWrapper,
        CoordinationLevel as CL,
        hconcat, vconcat,
    )
    from vitessce.widget_plugins import SpatialQueryPlugin

    label_key = sp.label_key
    spatial_key = sp.spatial_key

    has_expression = not sp.build_gene_index and sp.adata is not None

    # Build minimal AnnData: obs (cell type), obsm (coordinates), var (features)
    obs = sp.labels.to_frame()
    n_obs = obs.shape[0]
    var = pd.DataFrame({sp.feature_name: sp.genes}, index=sp.genes)
    n_var = len(sp.genes)

    if has_expression:
        X = sp.adata.X  # already normalized
    else:
        X = ssp.csr_matrix((n_obs, n_var))

    mini_adata = anndata.AnnData(X=X, obs=obs, var=var)
    mini_adata.obsm[spatial_key] = sp.spatial_pos

    if os.path.exists(zarr_path):
        warnings.warn(f"Zarr store already exists at '{zarr_path}'. Its contents will be overwritten.")
    else:
        os.makedirs(zarr_path)

    print(f"Writing zarr to {zarr_path} ...")
    mini_adata.write_zarr(zarr_path)

    plugin = SpatialQueryPlugin(
        mini_adata,
        spatial_key=spatial_key,
        label_key=label_key,
        feature_name=sp.feature_name,
        if_lognorm=False,
    )

    vc = VitessceConfig(schema_version="1.0.16", name="SpatialQuery")

    wrapper_kwargs = dict(
        adata_path=zarr_path,
        obs_set_paths=[f"obs/{label_key}"],
        obs_set_names=["Cell Type"],
        obs_spots_path=f"obsm/{spatial_key}",
    )
    if has_expression:
        wrapper_kwargs["obs_feature_matrix_path"] = "X"
        wrapper_kwargs["feature_labels_path"] = f"var/{sp.feature_name}"
        wrapper_kwargs["coordination_values"] = {"featureLabelsType": "Gene symbol"}

    dataset = vc.add_dataset("Query results").add_object(AnnDataWrapper(**wrapper_kwargs))

    spatial_view = vc.add_view("spatialBeta", dataset=dataset)
    lc_view = vc.add_view("layerControllerBeta", dataset=dataset)
    sets_view = vc.add_view("obsSets", dataset=dataset)
    sq_view = vc.add_view("spatialQuery", dataset=dataset)
    sq_heatmap_view = vc.add_view("spatialQueryHeatmap", dataset=dataset)

    initial_obs_set_selection = [
        ["Cell Type", ct] for ct in sp.labels.unique().tolist()
    ]

    linked_views = [spatial_view, lc_view, sets_view, sq_view]
    if has_expression:
        features_view = vc.add_view("featureList", dataset=dataset)
        linked_views.append(features_view)

    vc.link_views(
        linked_views,
        ["additionalObsSets", "obsSetColor", "obsSetSelection", "obsColorEncoding"],
        [plugin.additional_obs_sets, plugin.obs_set_color, initial_obs_set_selection, "cellSetSelection"],
    )

    vc.link_views_by_dict([spatial_view, lc_view], {
        "spotLayer": CL([{"obsType": "cell", "spatialSpotRadius": spatialSpotRadius}]),
    })

    if has_expression:
        vc.layout(
            vconcat(
                hconcat(spatial_view, vconcat(lc_view, features_view)),
                hconcat(sets_view, sq_heatmap_view, sq_view),
            )
        )
    else:
        vc.layout(
            vconcat(
                hconcat(spatial_view, lc_view),
                hconcat(sets_view, sq_heatmap_view, sq_view),
            )
        )

    return vc.widget(height=900, plugins=[plugin], remount_on_uid_change=False)
