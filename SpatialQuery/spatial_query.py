
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

from .scfind4sp import SCFind
import scanpy as sc
from . import spatial_utils, spatial_gene_covarying, plotting



class spatial_query:
    """
    Class for spatial query of single FOV

    Parameter
    ---------
    adata:
        AnnData object
    dataset:
        Dataset name, default is 'ST' of single FOV
    spatial_key:
        Key of spatial coordinates in adata.obsm, default is 'X_spatial'.
    label_key:
        Key of cell type label in adata.obs, default is 'predicted_label'
    leaf_size:
        Leaf size for KDTree, default is 10
    build_gene_index:
        Whether to build scfind index of expression data, default is False. If expression data is required for query,
        set this parameter to True
    feature_name:
        The label or key in the AnnData object's variables (var) that corresponds to the feature names. This is
        only used if build_gene_index is True
    if_lognorm:
        Whether to log normalize the expression data, default is True
    """
    def __init__(
        self,
        adata: AnnData,
        dataset: str = 'ST',
        spatial_key: str = 'X_spatial',
        label_key: str = 'predicted_label',
        leaf_size: int = 10,
        build_gene_index: bool = False,
        feature_name: str = None,
        if_lognorm: bool = True,
        ):
        if spatial_key not in adata.obsm.keys() or label_key not in adata.obs.keys():
            raise ValueError(f"The Anndata object must contain {spatial_key} in obsm and {label_key} in obs.")
        
        self.spatial_key = spatial_key

        # Standardize spatial position
        self.spatial_pos = np.array(adata.obsm[self.spatial_key])
        self.spatial_pos = spatial_utils._auto_normalize_spatial_coords(self.spatial_pos)
        
        self.dataset = dataset
        self.label_key = label_key
        self.labels = adata.obs[self.label_key]
        self.labels = self.labels.astype('category')
        self.kd_tree = KDTree(self.spatial_pos, leafsize=leaf_size)
        self.build_gene_index = build_gene_index
        
        self.adata = None
        self.genes = None
        self.index = None

        # filter features with NA
        valid_features = adata.var[feature_name].isna()
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

        Parameter
        ---------
        ct:
            Cell type name.
        k:
            Number of nearest neighbors.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        max_dist:
            Maximum distance for considering a cell as a neighbor.

        Return
        ------
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

        Parameter
        ---------
        ct:
            Cell type name.
        max_dist:
            Maximum distance for considering a cell as a neighbor.
        min_size:
            Minimum neighborhood size for each point to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.

        Return
        ------
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
                             motifs: Union[str, List[str]] = None,
                             k: int = 30,
                             min_support: float = 0.5,
                             max_dist: float = 20,
                             return_cellID: bool = False
                             ) -> pd.DataFrame:
        """
        Perform motif enrichment analysis using k-nearest neighbors (KNN).

        Parameter
        ---------
        ct:
            The cell type of the center cell.
        motifs:
            Specified motifs to be tested.
            If motifs=None, find the frequent patterns as motifs within the neighborhood of center cell type.
        k:
            Number of nearest neighbors to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        max_dist:
            Maximum distance for neighbors (default: 20).
        return_cellID:
            Indicate whether return cell IDs for each frequent pattern within the neighborhood of grid points.
            By defaults do not return cell ID.

        Return
        ------
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
                - corrected p-values: FDR-corrected p-values (only when multiple motifs tested)
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
                motifs = [motifs]

            labels_unique = self.labels.unique()
            motifs_exc = [m for m in motifs if m not in labels_unique]
            if len(motifs_exc) != 0:
                print(f"Found no {motifs_exc} in {self.label_key}. Ignoring them.")
            motifs = [m for m in motifs if m not in motifs_exc]
            motifs = [motifs]

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
            out_pd['corrected p-values'] = corrected_p_values
            out_pd['if_significant'] = if_rejected
            out_pd = out_pd.sort_values(by='corrected p-values', ignore_index=True)
            return out_pd

    def motif_enrichment_dist(self,
                              ct: str,
                              motifs: Union[str, List[str]] = None,
                              max_dist: float = 20,
                              min_size: int = 0,
                              min_support: float = 0.5,
                              return_cellID: bool = False,
                              ) -> DataFrame:
        """
        Perform motif enrichment analysis within a specified radius-based neighborhood.

        Parameter
        ---------
        ct:
            Cell type as the center cells.
        motifs:
            Specified motifs to be tested.
            If motifs=None, find the frequent patterns as motifs within the neighborhood of center cell type.
        max_dist:
            Maximum distance for considering a cell as a neighbor.
        min_size:
            Minimum neighborhood size for each point to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        return_cellID:
            Indicate whether return cell IDs for each motif within the neighborhood of central cell type.
            By defaults do not return cell ID.

        Return
        ------
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
                - corrected p-values: FDR-corrected p-values (only when multiple motifs tested)
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
                motifs = [motifs]

            labels_unique = self.labels.unique()
            motifs_exc = [m for m in motifs if m not in labels_unique]
            if len(motifs_exc) != 0:
                print(f"Found no {motifs_exc} in {self.label_key}. Ignoring them.")
            motifs = [m for m in motifs if m not in motifs_exc]
            motifs = [motifs]

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
            out_pd['corrected p-values'] = corrected_p_values
            out_pd['if_significant'] = if_rejected
            out_pd = out_pd.sort_values(by='corrected p-values', ignore_index=True)
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

        Parameter
        ---------
        max_dist:
            Maximum distance to consider a cell as a neighbor. Also used as grid spacing.
        min_size:
            Minimum neighborhood size for each grid point to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        if_display:
            Display the grid points with nearby frequent patterns if if_display=True.
        figsize:
            Tuple of figure size for the display plot.
        return_cellID:
            Indicate whether return cell IDs for each frequent pattern within the neighborhood of grid points.
            By default do not return cell ID.
        return_grid:
            Indicate whether return the grid points. By default, do not return grid points.
            If True, will return a tuple (fp_df, grid).

        Return
        ------
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

        Parameter
        ---------
        max_dist:
            Maximum distance to consider a cell as a neighbor.
        n_points:
            Number of random points to generate.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        min_size:
            Minimum neighborhood size for each random point to consider.
        if_display:
            Display the random points with nearby frequent patterns if if_display=True.
        figsize:
            Tuple of figure size for the display plot.
        return_cellID:
            Indicate whether return cell IDs for each frequent pattern within the neighborhood of random points.
            By default do not return cell ID.
        seed:
            Random seed for reproducibility.

        Return
        ------
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

        Paramaters
        ---------
        ind_group1: List of indices of cells in group 1.
        ind_group2: List of indices of cells in group 2.
        genes: List of gene names to query. If None, all genes will be used.
        min_fraction: The minimum fraction of cells that express a gene for it to be considered differentially expressed.
        method: The method to use for DE analysis. Please choose from fisher, t-test, or wilcoxon. If build_gene_index=True, only Fisher's exact test is supported.
        alpha: Significance threshold for adjusted p-values. If None, defaults to 0.1 when using Fisher's exact test and 0.05 otherwise.
        Return  
        ------
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
            out_df['adj_p_value'] = adjusted_pvals

            if alpha is None:
                alpha = 0.1

            results_df = out_df[out_df['adj_p_value'] < alpha]
            results_df.loc[:, 'de_in'] = np.where(
                (results_df['proportion_1'] >= results_df['proportion_2']),
                'group1',
                np.where(
                    (results_df['proportion_2'] > results_df['proportion_1']),
                    'group2',
                    None
                )
            )
            results_df = results_df[results_df['adj_p_value'] < alpha].sort_values('p_value').reset_index(drop=True)
        else:
            # Use adata.X directly for DE analysis
            if method == 'fisher':
                results_df = spatial_utils.de_genes_fisher(
                    self.adata, self.genes, ind_group1, ind_group2, genes, min_fraction, alpha
                )
            elif method == 't-test' or method == 'wilcoxon':
                results_df = spatial_utils.de_genes_scanpy(
                    self.adata, self.genes, ind_group1, ind_group2, genes, min_fraction, method=method, alpha=alpha
                )
            else:
                raise ValueError(f"Invalid method: {method}. Choose from 'fisher', 't-test', or 'wilcoxon'.")

        return results_df

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
        Compute gene-gene cross correlation between neighbor and non-neighbor motif cells. Only considers inter-cell-type interactions. 
        After finding neighbors using the full motif, removes all cells of the center cell type from both neighbor and
        non-neighbor groups. For Pearson correlation, uses shifted correlation (subtract cell type mean) to enable
        comparison across different niches/motifs.

        This function calculates cross correlation between gene expression in:
        1. Motif cells that are neighbors of center cell type (excluding center type cells in neighbor group)
        2. Motif cells that are NOT neighbors of center cell type (excluding center type cells)
        3. Neighboring cells of center cell type without nearby motif

        Parameter
        ---------
        ct:
            Cell type as the center cells.
        motif:
            Motif (names of cell types) to be analyzed. Include all cell types for neighbor finding.
        genes:
            List of genes to analyze. If None, all genes will be used.
        max_dist:
            Maximum distance for considering a cell as a neighbor. Use either max_dist or k.
        k:
            Number of nearest neighbors. Use either max_dist or k.
        min_size:
            Minimum neighborhood size for each center cell (only used when max_dist is specified).
        min_nonzero:
            Minimum number of non-zero expression values required for a gene to be included.
        alpha: 
            Significance threshold.

        Return
        ------
        Tuple[pd.DataFrame, dict]
            A tuple containing:

            results_df : DataFrame
                DataFrame with correlation results between neighbor and non-neighbor groups.
                Columns include:
                    - gene_center, gene_motif: gene pairs
                    - corr_neighbor: correlation in neighbor group
                    - corr_non_neighbor: correlation in non-neighbor group
                    - corr_diff_neighbor_vs_non: difference in correlation (neighbor - non_neighbor)
                    - p_value_test1: p-value for test1 (neighbor vs non-neighbor)
                    - delta_corr_test1: difference in correlation for test1
                    - corr_center_no_motif: correlation for centers without motif (NaN if not available)
                    - p_value_test2: p-value for test2 (neighbor vs no_motif)
                    - delta_corr_test2: difference in correlation for test2
                    - q_value_test1: FDR-corrected q-value for test1
                    - q_value_test2: FDR-corrected q-value for test2
                    - reject_test1_fdr: whether test1 passes FDR threshold
                    - reject_test2_fdr: whether test2 passes FDR threshold
                    - combined_score: combined significance score
                    - abs_combined_score: absolute value of combined score
                    - if_significant: whether both tests pass FDR threshold

            cell_groups : dict
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

                Note: Individual cell IDs can be extracted from pairs using np.unique() like:
                    - center_cells = np.unique(center_neighbor_motif_pair[:, 0])
                    - neighbor_cells = np.unique(center_neighbor_motif_pair[:, 1])
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

        Parameter
        ---------
        ct:
            Cell type as the center cells.
        motif:
            Motif (names of cell types) to be analyzed. Include all cell types for neighbor finding.
        genes:
            List of genes to analyze. If None, all genes will be used.
        max_dist:
            Maximum distance for considering a cell as a neighbor. Use either max_dist or k.
        k:
            Number of nearest neighbors. Use either max_dist or k.
        min_size:
            Minimum neighborhood size for each center cell (only used when max_dist is specified).
        min_nonzero:
            Minimum number of non-zero expression values required for a gene to be included.
        alpha:      
            Significance threshold.

        Return
        ------
        results_df : DataFrame
            DataFrame with correlation results for each cell type and gene pair.
            Columns include:
                - cell_type: the non-center cell type in motif
                - gene_center, gene_motif: gene pairs
                - corr_neighbor: correlation with neighboring cells of this type
                - corr_non_neighbor: correlation with distant cells of this type
                - corr_center_no_motif: correlation with neighbors when no motif present
                - p_value_test1: p-value for test1 (neighbor vs non-neighbor)
                - p_value_test2: p-value for test2 (neighbor vs no_motif)
                - q_value_test1: FDR-corrected q-value for test1
                - q_value_test2: FDR-corrected q-value for test2
                - delta_corr_test1: difference in correlation (neighbor - non_neighbor)
                - delta_corr_test2: difference in correlation (neighbor - no_motif)
                - reject_test1_fdr: whether test1 passes FDR threshold
                - reject_test2_fdr: whether test2 passes FDR threshold
                - combined_score: combined significance score
                - abs_combined_score: absolute value of combined score
                - if_significant: whether both tests pass FDR threshold
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

        Parameter
        ---------
        min_cells_label:
            Minimum number of cells required for a cell type to be displayed in the plot.
            Cell types with fewer cells will be excluded from the legend.
        title:
            Title of the figure.
        figsize:
            Figure size as (width, height) tuple.
        save_path:
            Path to save the figure. If None, the figure will not be saved.

        Return
        ------
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

        Parameter
        ---------
        motif:
            Motif (names of cell types) to be colored
        max_dist:
            Spacing distance for building grid.
        figsize:
            Figure size.
        save_path:
            Path to save the figure.
            If None, the figure will not be saved.

        Return
        ------
        A figure.
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

        Parameter
        ---------
        motif:
            Motif (names of cell types) to be colored
        max_dist:
            Radius for neighborhood search.
        n_points:
            Number of random points to generate.
        figsize:
            Figure size.
        seed:
            Set random seed for reproducible.
        save_path:
            Path to save the figure.
            If None, the figure will not be saved.

        Return
        ------
        A figure.
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


        Parameter
        ---------
        ct:
            Cell type as the center cells.
        motif:
            Motif (names of cell types) to be colored.
        max_dist:
            Spacing distance for building grid. Make sure using the same value as that in find_patterns_grid.
        figsize:
            Figure size.
        save_path:
            Path to save the figure.
            If None, the figure will not be saved.

        Return
        ------
        A figure.
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

        Parameter
        --------
        center:
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
        figsize: tuple
            Figure size.

        save_path:
            Path to save the figure.
            If None, the figure will not be saved.

        Return
        ------
        A figure.
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

        Parameter
        ---------
        fp_df:
            DataFrame containing frequent pattern results with two columns:
                - support: frequency of the pattern (proportion of points with this pattern)
                - itemsets: frozenset of cell types in the frequent pattern
            This is typically the output from find_fp_knn or find_fp_dist methods.
        figsize:
            Figure size, default is (7, 5)
        save_path:
            Path to save the figure. If None, the figure will not be saved.
        title:
            Figure title. If None, will use a default title.
        cmap:
            Colormap for the heatmap, default is 'GnBu'

        Return
        ------
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

        Parameter
        ---------
        enrich_df:
            Output DataFrame from motif_enrichment_dist or motif_enrichment_knn
        figsize:
            Figure size, default is (7, 5)
        save_path:
            Path to save the figure. If None, the figure will not be saved.
        title:
            Figure title. If None, will use a default title based on center cell type.
        cmap:
            Colormap for the heatmap, default is 'GnBu'

        Return
        ------
        A figure showing the heatmap of motif cell type distribution.
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
        Plot a heatmap showing the cross-varying gene pairs.
        
        Parameter
        ---------
        gene_pair_df:
            Output DataFrame from compute_gene_gene_correlation or compute_gene_gene_correlation_by_type.
        figsize:
            Figure size, default is (7, 5)
        save_path:
            Path to save the figure. If None, the figure will not be saved.
        title:
            Figure title. If None, will use a default title based on center cell type.
        cmap:
            Colormap for the heatmap, default is 'GnBu'
        
        Return
        ------
        A figure showing the heatmap of motif cell type distribution.
        """
        return plotting.plot_gene_pair_heatmap(gene_pair_df=gene_pair_df, figsize=figsize,
                                                       save_path=save_path, title=title, cmap=cmap)
