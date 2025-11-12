from collections import Counter
from typing import List, Union, Optional, Literal

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
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as scipy_stats
import matplotlib.cm as cm
from time import time

from .scfind4sp import SCFind
import scanpy as sc
from . import spatial_utils



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
    max_radius:
        The upper limit of neighborhood radius, default is 500
    n_split:
        The number of splits in each axis for spatial grid to speed up query, default is 10
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
        max_radius: float = 20,
        n_split: int = 10,
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
        self.max_radius = max_radius
        self.labels = adata.obs[self.label_key]
        self.labels = self.labels.astype('category')
        self.kd_tree = KDTree(self.spatial_pos, leafsize=leaf_size)
        self.overlap_radius = max_radius  # the upper limit of radius in case missing cells with large radius of query
        self.n_split = n_split
        self.grid_cell_types, self.grid_indices = spatial_utils.initialize_grids(
            self.spatial_pos, self.labels, self.n_split, self.overlap_radius
        )
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


    def find_fp_knn(self,
                    ct: str,
                    k: int = 30,
                    min_support: float = 0.5,
                    max_dist: float = 200,
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
                     max_dist: float = 100,
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
        max_dist = min(max_dist, self.max_radius)

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
                             max_dist: float = 200,
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
            Maximum distance for neighbors (default: 200).
        return_cellID:
            Indicate whether return cell IDs for each frequent pattern within the neighborhood of grid points.
            By defaults do not return cell ID.

        Return
        ------
        pd.Dataframe containing the cell type name, motifs, number of motifs nearby given cell type,
        number of spots of cell type, number of motifs in single FOV, p value of hypergeometric distribution.
        """
        if ct not in self.labels.unique():
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        max_dist = min(max_dist, self.max_radius)

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
            raise ValueError("No frequent patterns were found. Please lower min_support value.")

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
                              max_dist: float = 100,
                              min_size: int = 0,
                              min_support: float = 0.5,
                              max_ns: int = 100,
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
        max_ns:
            Maximum number of neighborhood size for each point.
        return_cellID:
            Indicate whether return cell IDs for each motif within the neighborhood of central cell type.
            By defaults do not return cell ID.
        Returns
        -------
        Tuple containing counts and statistical measures.
        """
        if ct not in self.labels.unique():
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        out = []
        max_dist = min(max_dist, self.max_radius)
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

        label_encoder = LabelEncoder()
        int_labels = label_encoder.fit_transform(np.array(self.labels))
        int_ct = label_encoder.transform(np.array(ct, dtype=object, ndmin=1))

        cinds = np.where(self.labels == ct)[0]

        num_cells = len(self.spatial_pos)
        num_types = len(label_encoder.classes_)

        if return_cellID:
            idxs_all = self.kd_tree.query_ball_point(
                self.spatial_pos,
                r=max_dist,
                return_sorted=False,
                workers=-1,
            )
            idxs_all_filter = [np.array(ids)[np.array(ids) != i] for i, ids in enumerate(idxs_all)]
            flat_neighbors_all = np.concatenate(idxs_all_filter)
            row_indices_all = np.repeat(np.arange(num_cells), [len(neigh) for neigh in idxs_all_filter])
            neighbor_labels_all = int_labels[flat_neighbors_all]
            mask_all = int_labels == int_ct

        for motif in motifs:
            motif = list(motif) if not isinstance(motif, list) else motif
            sort_motif = sorted(motif)

            _, matching_cells_indices = spatial_utils.query_pattern(
                motif, self.grid_cell_types, self.grid_indices
            )
            if not matching_cells_indices:
                # if matching_cells_indices is empty, it indicates no motif are grouped together within upper limit of radius (500)
                continue
            matching_cells_indices = np.concatenate([t for t in matching_cells_indices.values()])
            matching_cells_indices = np.unique(matching_cells_indices)
            # print(f"number of cells skipped: {len(matching_cells_indices)}")
            print(f"proportion of cells searched: {len(matching_cells_indices) / len(self.spatial_pos)}")
            idxs_in_grids = self.kd_tree.query_ball_point(
                self.spatial_pos[matching_cells_indices],
                r=max_dist,
                return_sorted=True,
                workers=-1
            )

            # using numpy
            int_motifs = label_encoder.transform(np.array(motif))

            # filter center out of neighbors
            idxs_filter = [np.array(ids)[np.array(ids) != i][:min(max_ns, len(ids))] for i, ids in
                           zip(matching_cells_indices, idxs_in_grids)]

            flat_neighbors = np.concatenate(idxs_filter)
            row_indices = np.repeat(np.arange(len(matching_cells_indices)), [len(neigh) for neigh in idxs_filter])
            neighbor_labels = int_labels[flat_neighbors]

            neighbor_matrix = np.zeros((len(matching_cells_indices), num_types), dtype=int)
            np.add.at(neighbor_matrix, (row_indices, neighbor_labels), 1)

            mask = int_labels[matching_cells_indices] == int_ct
            n_motif_ct = np.sum(np.all(neighbor_matrix[mask][:, int_motifs] > 0, axis=1))
            n_motif_labels = np.sum(np.all(neighbor_matrix[:, int_motifs] > 0, axis=1))

            n_ct = len(cinds)
            if ct in motif:
                n_ct = round(n_ct / motif.count(ct))

            hyge = hypergeom(M=len(self.labels), n=n_ct, N=n_motif_labels)
            motif_out = {'center': ct, 'motifs': sort_motif, 'n_center_motif': n_motif_ct,
                         'n_center': n_ct, 'n_motif': n_motif_labels, 'expectation': hyge.mean(),
                         'p-values': hyge.sf(n_motif_ct)}

            if return_cellID:
                neighbor_matrix_all = np.zeros((num_cells, num_types), dtype=int)
                np.add.at(neighbor_matrix_all, (row_indices_all, neighbor_labels_all), 1)
                inds_all = np.where(np.all(neighbor_matrix_all[mask_all][:, int_motifs] > 0, axis=1))[0]
                cind_with_motif = np.array([cinds[i] for i in inds_all])
                motif_mask = np.isin(np.array(self.labels), motif)
                all_neighbors = np.concatenate([idxs_all_filter[i] for i in cind_with_motif])
                valid_neighbors = all_neighbors[motif_mask[all_neighbors]]
                id_motif_celltype = set(valid_neighbors)
                motif_out['neighbor_id'] = np.array(list(id_motif_celltype))
                motif_out['center_id'] = np.array(cind_with_motif)

            out.append(motif_out)

        out_pd = pd.DataFrame(out)

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
                           max_dist: float = 100,
                           min_size: int = 0,
                           min_support: float = 0.5,
                           if_display: bool = True,
                           fig_size: tuple = (10, 5),
                           return_cellID: bool = False,
                           return_grid: bool = False,
                           ):
        """
        Create a grid and use it to find surrounding patterns in spatial data.

        Parameter
        ---------
        max_dist:
            Maximum distance to consider a cell as a neighbor.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern
        min_size:
            Additional parameters for pattern finding.
        if_display:
            Display the grid points with nearby frequent patterns if if_display=True.
        fig_size:
            Tuple of figure size.
        return_cellID:
            Indicate whether return cell IDs for each frequent pattern within the neighborhood of grid points.
            By defaults do not return cell ID.
        return_grid:
            Indicate whether return the grid points. By default, do not return grid points.
            If true, will return a tuple (fp_tree, grid)

        Return
        ------
        fp_tree:
            Frequent patterns
        """

        max_dist = min(max_dist, self.max_radius)
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
            fig, ax = plt.subplots(figsize=fig_size)
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
                           max_dist: float = 100,
                           n_points: int = 1000,
                           min_support: float = 0.5,
                           min_size: int = 0,
                           if_display: bool = True,
                           fig_size: tuple = (10, 5),
                           return_cellID: bool = False,
                           seed: int = 2023) -> DataFrame:
        """
        Randomly generate points and use them to find surrounding patterns in spatial data.

        Parameter
        ---------
        if_knn:
            Use k-nearest neighbors or points within max_dist distance as neighborhood.
        k:
            Number of nearest neighbors. If if_knn=True, parameter k is used.
        max_dist:
            Maximum distance to consider a cell as a neighbor. If if_knn=False, parameter max_dist is used.
        n_points:
            Number of random points to generate.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        min_size:
            Additional parameters for pattern finding.
        if_display:
            Display the grid points with nearby frequent patterns if if_display=True.
        fig_size:
            Tuple of figure size.
        return_cellID:
            Indicate whether return cell IDs for each frequent pattern within the neighborhood of grid points.
            By defaults do not return cell ID.
        seed:
            Set random seed for reproducible.

        Return
        ------
        Results from the pattern finding function.
        """

        max_dist = min(max_dist, self.max_radius)
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
            fig, ax = plt.subplots(figsize=fig_size)
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
        Return  
        ------
        pd.DataFrame containing the differentially expressed genes between the two groups.
        """
        if self.build_gene_index:
            # Use scfind index for DE analysis with Fisher's exact test.
            if genes is None:
                genes = self.index.scfindGenes

            print(f'Using scfind index for DE analysis with {method} test.')

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
            results_df = out_df[out_df['adj_p_value'] < 0.05]
            results_df.loc[:, 'de_in'] = np.where(
                (results_df['proportion_1'] >= results_df['proportion_2']),
                'group1',
                np.where(
                    (results_df['proportion_2'] > results_df['proportion_1']),
                    'group2',
                    None
                )
            )
            results_df = results_df[results_df['adj_p_value'] < 0.05].sort_values('p_value').reset_index(drop=True)
        else:
            # Use adata.X directly for DE analysis
            if method == 'fisher':
                results_df = spatial_utils.de_genes_fisher(
                    self.adata, self.genes, ind_group1, ind_group2, genes, min_fraction
                )
            elif method == 't-test' or method == 'wilcoxon':
                results_df = spatial_utils.de_genes_scanpy(
                    self.adata, self.genes, ind_group1, ind_group2, genes, min_fraction, method=method
                )
            else:
                raise ValueError(f"Invalid method: {method}. Choose from 'fisher', 't-test', or 'wilcoxon'.")

        return results_df


    def plot_fov(self,
                 min_cells_label: int = 50,
                 title: str = 'Spatial distribution of cell types',
                 fig_size: tuple = (10, 5),
                 save_path: Optional[str] = None,
                 ):
        """
        Plot the cell type distribution of single fov.

        Parameter
        --------
        min_cells_label:
            Minimum number of points in each cell type to display.
        title:
            Figure title.
        fig_size:
            Figure size parameter.

        save_path:
            Path to save the figure.
            If None, the figure will not be saved.

        Return
        ------
        A figure.
        """
        # Ensure that 'spatial' and label_key are present in the Anndata object

        cell_type_counts = self.labels.value_counts()
        n_colors = sum(cell_type_counts >= min_cells_label)
        colors = sns.color_palette('hsv', n_colors)

        color_counter = 0
        fig, ax = plt.subplots(figsize=fig_size)

        # Iterate over each cell type
        for cell_type in sorted(self.labels.unique()):
            # Filter data for each cell type
            index = self.labels == cell_type
            index = np.where(index)[0]
            # data = self.labels[self.labels == cell_type].index
            # Check if the cell type count is above the threshold
            if cell_type_counts[cell_type] >= min_cells_label:
                ax.scatter(self.spatial_pos[index, 0], self.spatial_pos[index, 1],
                           label=cell_type, color=colors[color_counter], s=1)
                color_counter += 1
            else:
                ax.scatter(self.spatial_pos[index, 0], self.spatial_pos[index, 1],
                           color='grey', s=1)

        handles, labels = ax.get_legend_handles_labels()

        # Modify labels to include count values
        new_labels = [f'{label} ({cell_type_counts[label]})' for label in labels]

        # Create new legend
        ax.legend(handles, new_labels, loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)
        # ax.legend(handles, new_labels, loc='lower center', bbox_to_anchor=(1, 0.5), markerscale=4)

        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)

        plt.xlabel('Spatial X')
        plt.ylabel('Spatial Y')
        plt.title(title)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        # Adjust layout to prevent clipping of ylabel and accommodate the legend
        plt.tight_layout(rect=[0, 0, 1.1, 1])
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_motif_grid(self,
                        motif: Union[str, List[str]],
                        fig_size: tuple = (10, 5),
                        max_dist: float = 100,
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
        fig_size:
            Figure size.
        save_path:
            Path to save the figure.
            If None, the figure will not be saved.

        Return
        ------
        A figure.
        """
        if isinstance(motif, str):
            motif = [motif]

        max_dist = min(max_dist, self.max_radius)

        labels_unique = self.labels.unique()
        motif_exc = [m for m in motif if m not in labels_unique]
        if len(motif_exc) != 0:
            print(f"Found no {motif_exc} in {self.label_key}. Ignoring them.")
        motif = [m for m in motif if m not in motif_exc]

        # Build mesh
        xmax, ymax = np.max(self.spatial_pos, axis=0)
        xmin, ymin = np.min(self.spatial_pos, axis=0)
        x_grid = np.arange(xmin - max_dist, xmax + max_dist, max_dist)
        y_grid = np.arange(ymin - max_dist, ymax + max_dist, max_dist)
        grid = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)

        # self.build_fptree_dist returns valid_idxs () instead of all the idxs,
        # so recalculate the idxs directly using self.kd_tree.query_ball_point
        idxs = self.kd_tree.query_ball_point(grid, r=max_dist, return_sorted=False, workers=-1)

        # Locate the index of grid points acting as centers with motif nearby
        id_center = []
        for i, idx in enumerate(idxs):
            ns = [self.labels[id] for id in idx]
            if spatial_utils.has_motif(neighbors=motif, labels=ns):
                id_center.append(i)

        # Locate the index of cell types contained in motif in the
        # neighborhood of above grid points with motif nearby
        id_motif_celltype = set()
        for i in id_center:
            idx = idxs[i]
            for cell_id in idx:
                if self.labels[cell_id] in motif:
                    id_motif_celltype.add(cell_id)

        # Plot above spots and center grid points
        # Set color map using motif cell types
        motif_unique = sorted(set(motif))
        n_colors = len(motif_unique)
        colors = sns.color_palette('hsv', n_colors)
        color_map = {ct: col for ct, col in zip(motif_unique, colors)}

        motif_spot_pos = self.spatial_pos[list(id_motif_celltype), :]
        motif_spot_label = self.labels[list(id_motif_celltype)]
        fig, ax = plt.subplots(figsize=fig_size)
        # Plotting the grid lines
        for x in x_grid:
            ax.axvline(x, color='lightgray', linestyle='--', lw=0.5)

        for y in y_grid:
            ax.axhline(y, color='lightgray', linestyle='--', lw=0.5)
        ax.scatter(grid[id_center, 0], grid[id_center, 1], label='Grid Points',
                   edgecolors='red', facecolors='none', s=8)

        # Plotting other spots as background
        bg_index = [i for i, _ in enumerate(self.labels) if
                    i not in id_motif_celltype]  # the other spots are colored as background
        # bg_adata = self.adata[bg_index, :]
        bg_pos = self.spatial_pos[bg_index, :]
        ax.scatter(bg_pos[:, 0],
                   bg_pos[:, 1],
                   color='darkgrey', s=1)

        for ct in motif_unique:
            ct_ind = motif_spot_label == ct
            ax.scatter(motif_spot_pos[ct_ind, 0],
                       motif_spot_pos[ct_ind, 1],
                       label=ct, color=color_map[ct], s=1)

        ax.set_xlim([xmin - max_dist, xmax + max_dist])
        ax.set_ylim([ymin - max_dist, ymax + max_dist])
        ax.legend(title='motif', loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)
        # ax.legend(title='motif', loc='lower center', bbox_to_anchor=(0, 0.), markerscale=4)
        plt.xlabel('Spatial X')
        plt.ylabel('Spatial Y')
        plt.title('Spatial distribution of frequent patterns')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout(rect=[0, 0, 1.1, 1])
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_motif_rand(self,
                        motif: Union[str, List[str]],
                        max_dist: float = 100,
                        n_points: int = 1000,
                        fig_size: tuple = (10, 5),
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
        fig_size:
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
        if isinstance(motif, str):
            motif = [motif]

        max_dist = min(max_dist, self.max_radius)

        labels_unique = self.labels.unique()
        motif_exc = [m for m in motif if m not in labels_unique]
        if len(motif_exc) != 0:
            print(f"Found no {motif_exc} in {self.label_key}. Ignoring them.")
        motif = [m for m in motif if m not in motif_exc]

        # Random sample points
        xmax, ymax = np.max(self.spatial_pos, axis=0)
        xmin, ymin = np.min(self.spatial_pos, axis=0)
        np.random.seed(seed)
        pos = np.column_stack((np.random.rand(n_points) * (xmax - xmin) + xmin,
                               np.random.rand(n_points) * (ymax - ymin) + ymin))

        idxs = self.kd_tree.query_ball_point(pos, r=max_dist, return_sorted=False, workers=-1)

        # Locate the index of random points acting as centers with motif nearby
        id_center = []
        for i, idx in enumerate(idxs):
            ns = [self.labels[id] for id in idx]
            if spatial_utils.has_motif(neighbors=motif, labels=ns):
                id_center.append(i)

        # Locate the index of cell types contained in motif in the
        # neighborhood of above random points with motif nearby
        id_motif_celltype = set()
        for i in id_center:
            idx = idxs[i]
            for cell_id in idx:
                if self.labels[cell_id] in motif:
                    id_motif_celltype.add(cell_id)

        # Plot above spots and center random points
        # Set color map using motif cell types
        motif_unique = sorted(set(motif))
        n_colors = len(motif_unique)
        colors = sns.color_palette('hsv', n_colors)
        color_map = {ct: col for ct, col in zip(motif_unique, colors)}

        motif_spot_pos = self.spatial_pos[list(id_motif_celltype), :]
        motif_spot_label = self.labels[list(id_motif_celltype)]
        fig, ax = plt.subplots(figsize=fig_size)
        ax.scatter(pos[id_center, 0], pos[id_center, 1], label='Random Sampling Points',
                   edgecolors='red', facecolors='none', s=8)

        # Plotting other spots as background
        bg_index = [i for i, _ in enumerate(self.labels) if
                    i not in id_motif_celltype]  # the other spots are colored as background
        bg_adata = self.spatial_pos[bg_index, :]
        ax.scatter(bg_adata[:, 0],
                   bg_adata[:, 1],
                   color='darkgrey', s=1)
        for ct in motif_unique:
            ct_ind = motif_spot_label == ct
            ax.scatter(motif_spot_pos[ct_ind, 0],
                       motif_spot_pos[ct_ind, 1],
                       label=ct, color=color_map[ct], s=1)

        ax.set_xlim([xmin - max_dist, xmax + max_dist])
        ax.set_ylim([ymin - max_dist, ymax + max_dist])
        ax.legend(title='motif', loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)
        plt.xlabel('Spatial X')
        plt.ylabel('Spatial Y')
        plt.title('Spatial distribution of frequent patterns')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout(rect=[0, 0, 1.1, 1])
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def get_motif_neighbor_cells(self,
                                 ct: str,
                                 motif: Union[str, List[str]],
                                 max_dist: Optional[float] = None,
                                 k: Optional[int] = None,
                                 min_size: int = 0,
                                 ) -> dict:
        """
        Get cell IDs of motif cells that are neighbors of the center cell type.
        Similar to motif_enrichment_* with return_cellID=True, but only returns cell IDs 
        and move center type of motifs to center cells for gene-gene covarying analysis.

        For kNN: filters out neighbors beyond min(max_dist, self.max_radius).
        For dist: filters out center cells with fewer than min_size neighbors.

        If center type (ct) is in motif, motif cells of center type are also included in center_id.

        Parameter
        ---------
        ct:
            Cell type as the center cells.
        motif:
            Motif (names of cell types) to be identified.
        max_dist:
            Maximum distance for considering a cell as a neighbor. Use either max_dist or k.
        k:
            Number of nearest neighbors. Use either max_dist or k.
        min_size:
            Minimum neighborhood size for each center cell (only used when max_dist is specified).

        Return
        ------
        dict with keys:
            'center_neighbor_pairs': array of shape (n_pairs, 2) with (center_idx, neighbor_idx) pairs
            'ct_in_motif': bool, whether center type is in motif list

        Note: center_id and neighbor_id can be extracted from pairs:
            center_id = np.unique(pairs[:, 0])
            neighbor_id = np.unique(pairs[:, 1])
        Special handling: if ct_in_motif is True, center cells of motif type that appear
        as neighbors should be moved to center group. User should apply this logic after extraction.
        """
        if (max_dist is None and k is None) or (max_dist is not None and k is not None):
            raise ValueError("Please specify either max_dist or k, but not both.")

        if isinstance(motif, str):
            motif = [motif]

        motif_exc = [m for m in motif if m not in self.labels.unique()]
        if len(motif_exc) != 0:
            print(f"Found no {motif_exc} in {self.label_key}. Ignoring them.")
        motif = [m for m in motif if m not in motif_exc]

        if ct not in self.labels.unique():
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        cinds = np.where(self.labels == ct)[0]

        # Check if ct is in motif - if so, we'll add those motif cells to center_id later
        ct_in_motif = ct in motif

        motif_mask = np.isin(np.array(self.labels), motif)

        if max_dist is not None:
            # Distance-based neighbors - only query for center cells
            max_dist = min(max_dist, self.max_radius)
            idxs_centers = self.kd_tree.query_ball_point(
                self.spatial_pos[cinds],
                r=max_dist,
                return_sorted=False,
                workers=-1,
            )

            # Process all centers in one loop: remove self, filter by min_size, check motif, and build pairs
            center_neighbor_pairs = []

            for i, cind in enumerate(cinds):
                # Remove self from neighbors
                neighbors = np.array([n for n in idxs_centers[i] if n != cind])

                # Apply min_size filter
                if len(neighbors) >= min_size:

                    # Check if all motif types are present in neighbors
                    neighbor_labels = self.labels[neighbors]
                    neighbors_unique = neighbor_labels.unique().tolist()
                    has_all_motifs = all([m in neighbors_unique for m in motif])

                    if has_all_motifs:
                        # Get motif neighbors for this center
                        motif_neighbors = neighbors[motif_mask[neighbors]]

                        # Build pairs in the same loop (vectorized)
                        if ct_in_motif:
                            center_type_neighs = motif_neighbors[self.labels[motif_neighbors] == ct]
                            non_center_type_neighs = motif_neighbors[self.labels[motif_neighbors] != ct]

                            if len(non_center_type_neighs) > 0:
                                # Pair original center with non-center type neighbors (vectorized)
                                pairs_center = np.column_stack([
                                    np.repeat(cind, len(non_center_type_neighs)),
                                    non_center_type_neighs
                                ])
                                center_neighbor_pairs.extend(pairs_center)

                                # For center type neighbors, pair them with other non-center type neighbors (vectorized)
                                if len(center_type_neighs) > 0:
                                    # Create all combinations: each center_type_neigh with each non_center_type_neigh
                                    pairs_ct = np.column_stack([
                                        np.repeat(center_type_neighs, len(non_center_type_neighs)),
                                        np.tile(non_center_type_neighs, len(center_type_neighs))
                                    ])
                                    center_neighbor_pairs.extend(pairs_ct)
                        else:
                            # Pair center with all motif neighbors (vectorized)
                            if len(motif_neighbors) > 0:
                                pairs = np.column_stack([
                                    np.repeat(cind, len(motif_neighbors)),
                                    motif_neighbors
                                ])
                                center_neighbor_pairs.extend(pairs)
        else:
            # KNN-based neighbors - only query for center cells
            dists, idxs = self.kd_tree.query(self.spatial_pos[cinds], k=k + 1, workers=-1)

            # Apply distance cutoff
            if max_dist is None:
                max_dist = self.max_radius
            max_dist = min(max_dist, self.max_radius)

            valid_neighbors = dists[:, 1:] <= max_dist

            # Process all centers in one loop and build pairs
            center_neighbor_pairs = []

            for i, cind in enumerate(cinds):
                valid_neighs = idxs[i, 1:][valid_neighbors[i, :]]

                # Check if all motif types are present in neighbors
                neighbor_labels = self.labels[valid_neighs]
                neighbors_unique = neighbor_labels.unique().tolist()
                has_all_motifs = all([m in neighbors_unique for m in motif])

                if has_all_motifs:
                    # Get motif neighbors for this center
                    motif_neighbors = valid_neighs[motif_mask[valid_neighs]]

                    # Build pairs in the same loop (vectorized)
                    if ct_in_motif:
                        center_type_neighs = motif_neighbors[self.labels[motif_neighbors] == ct]
                        non_center_type_neighs = motif_neighbors[self.labels[motif_neighbors] != ct]

                        if len(non_center_type_neighs) > 0:
                            # Pair original center with non-center type neighbors (vectorized)
                            pairs_center = np.column_stack([
                                np.repeat(cind, len(non_center_type_neighs)),
                                non_center_type_neighs
                            ])
                            center_neighbor_pairs.extend(pairs_center)

                            # For center type neighbors, pair them with other non-center type neighbors (vectorized)
                            if len(center_type_neighs) > 0:
                                # Create all combinations: each center_type_neigh with each non_center_type_neigh
                                pairs_ct = np.column_stack([
                                    np.repeat(center_type_neighs, len(non_center_type_neighs)),
                                    np.tile(non_center_type_neighs, len(center_type_neighs))
                                ])
                                center_neighbor_pairs.extend(pairs_ct)
                    else:
                        # Pair center with all motif neighbors (vectorized)
                        if len(motif_neighbors) > 0:
                            pairs = np.column_stack([
                                np.repeat(cind, len(motif_neighbors)),
                                motif_neighbors
                            ])
                            center_neighbor_pairs.extend(pairs)

        center_neighbor_pairs = np.array(center_neighbor_pairs) if len(center_neighbor_pairs) > 0 else np.array([]).reshape(0, 2)

        # Remove duplicate pairs (can occur when center-type cells are neighbors of each other)
        if len(center_neighbor_pairs) > 0:
            n_pairs_before = len(center_neighbor_pairs)
            center_neighbor_pairs = np.unique(center_neighbor_pairs, axis=0)
            n_duplicates = n_pairs_before - len(center_neighbor_pairs)
            if n_duplicates > 0:
                print(f"Removed {n_duplicates} duplicate pairs ({n_duplicates/n_pairs_before*100:.1f}%)")
        else:
            raise ValueError("Error: No motif-neighbor pairs found! Please adjust neighborhood size.")


        # Return only pairs and ct_in_motif flag
        # User can extract center_id and neighbor_id from pairs as needed
        return {
            'center_neighbor_pairs': center_neighbor_pairs,
            'ct_in_motif': ct_in_motif
        }

    def get_all_neighbor_cells(self,
                               ct: str,
                               max_dist: Optional[float] = None,
                               k: Optional[int] = None,
                               min_size: int = 0,
                               exclude_centers: Optional[np.ndarray] = None,
                               exclude_neighbors: Optional[np.ndarray] = None,
                               ) -> dict:
        """
        Get all neighbor cells (not limited to motif) for given center cell type excluding center cells in exclude_centers.
        Similar to get_motif_neighbor_cells but returns ALL neighbors regardless of cell type.
        And only returns neighbors that are different from center cell type.

        Parameter
        ---------
        ct:
            Cell type as the center cells.
        max_dist:
            Maximum distance for considering a cell as a neighbor. Use either max_dist or k.
        k:
            Number of nearest neighbors. Use either max_dist or k.
        min_size:
            Minimum neighborhood size for each center cell (only used when max_dist is specified).
        exclude_centers:
            Array of center cell IDs to exclude from the search. Only search neighbors for
            centers NOT in this array.
        exclude_neighbors:
            Array of neighbor cell IDs to exclude from the results. Typically used to exclude
            motif neighbors from get_motif_neighbor_cells.

        Return
        ------
        dict with keys:
            'center_neighbor_pairs': array of shape (n_pairs, 2) with (center_idx, neighbor_idx) pairs

        Note: center_id and neighbor_id can be extracted from pairs:
            center_id = np.unique(pairs[:, 0])
            neighbor_id = np.unique(pairs[:, 1])
        """
        if (max_dist is None and k is None) or (max_dist is not None and k is not None):
            raise ValueError("Please specify either max_dist or k, but not both.")

        if ct not in self.labels.unique():
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        cinds = np.where(self.labels == ct)[0]

        # Exclude specified centers if provided
        if exclude_centers is not None:
            cinds = np.setdiff1d(cinds, exclude_centers)

        # Build pairs directly
        center_neighbor_pairs = []

        if max_dist is not None:
            # Distance-based neighbors - only query for center cells
            max_dist = min(max_dist, self.max_radius)
            idxs_centers = self.kd_tree.query_ball_point(
                self.spatial_pos[cinds],
                r=max_dist,
                return_sorted=False,
                workers=-1,
            )

            # Process all centers in one loop and build pairs
            for i, cind in enumerate(cinds):
                # Remove self from neighbors
                neighbors = np.array([n for n in idxs_centers[i] if n != cind])

                # Filter by min_size
                if len(neighbors) >= min_size:
                    # Exclude center type from neighbors
                    valid_neighbors = neighbors[self.labels[neighbors] != ct]

                    # Exclude specified neighbors (e.g., motif neighbors)
                    if exclude_neighbors is not None and len(exclude_neighbors) > 0:
                        valid_neighbors = np.setdiff1d(valid_neighbors, exclude_neighbors)

                    if len(valid_neighbors) > 0:
                        # Build pairs vectorized
                        pairs = np.column_stack([
                            np.repeat(cind, len(valid_neighbors)),
                            valid_neighbors
                        ])
                        center_neighbor_pairs.extend(pairs)

        else:
            # KNN-based neighbors - only query for center cells
            dists, idxs = self.kd_tree.query(self.spatial_pos[cinds], k=k + 1, workers=-1)

            # Apply distance cutoff
            if max_dist is None:
                max_dist = self.max_radius
            max_dist = min(max_dist, self.max_radius)

            valid_neighbors = dists[:, 1:] <= max_dist

            # Process all centers in one loop and build pairs
            for i, cind in enumerate(cinds):
                valid_neighs = idxs[i, 1:][valid_neighbors[i, :]]

                # Exclude center type from neighbors
                valid_neighs_filtered = valid_neighs[self.labels[valid_neighs] != ct]

                # Exclude specified neighbors (e.g., motif neighbors)
                if exclude_neighbors is not None and len(exclude_neighbors) > 0:
                    valid_neighs_filtered = np.setdiff1d(valid_neighs_filtered, exclude_neighbors)

                if len(valid_neighs_filtered) > 0:
                    # Build pairs vectorized
                    pairs = np.column_stack([
                        np.repeat(cind, len(valid_neighs_filtered)),
                        valid_neighs_filtered
                    ])
                    center_neighbor_pairs.extend(pairs)

        center_neighbor_pairs = np.array(center_neighbor_pairs) if len(center_neighbor_pairs) > 0 else np.array([]).reshape(0, 2)

        if len(center_neighbor_pairs) > 0:
            center_neighbor_pairs = np.unique(center_neighbor_pairs, axis=0)
        else:
            print("Warning: No motif-neighbor pairs found!")

        # Return only pairs - user can extract center_id and neighbor_id as needed
        return {
            'center_neighbor_pairs': center_neighbor_pairs
        }

    def compute_gene_gene_correlation(self,
                                      ct: str,
                                      motif: Union[str, List[str]],
                                      genes: Optional[Union[str, List[str]]] = None,
                                      max_dist: Optional[float] = None,
                                      k: Optional[int] = None,
                                      min_size: int = 0,
                                      min_nonzero: int = 10,
                                      ) -> pd.DataFrame:
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

        Return
        ------
        results_df : DataFrame
            DataFrame with correlation results between neighbor and non-neighbor groups.
            Columns include:
                - gene_center, gene_motif: gene pairs
                - corr_neighbor: correlation in neighbor group
                - corr_non_neighbor: correlation in non-neighbor group
                - corr_diff: difference in correlation (neighbor - non_neighbor)
                - n_neighbor: number of cells in neighbor group (after removing center type)
                - n_non_neighbor: number of cells in non-neighbor group (after removing center type)

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
        if self.adata is None:
            raise ValueError("Expression data (adata) is not available. Please set build_gene_index=False when initializing spatial_query.")
        
        motif = motif if isinstance(motif, list) else [motif]

        # Get neighbor and non-neighbor cell IDs plus paired mappings (using original motif)
        neighbor_result = self.get_motif_neighbor_cells(ct=ct, motif=motif, max_dist=max_dist, k=k, min_size=min_size)

        # Extract paired data and derive cell IDs from pairs
        center_neighbor_pairs = neighbor_result['center_neighbor_pairs']
        ct_in_motif = neighbor_result['ct_in_motif']

        # Extract unique center and neighbor cells from pairs
        # Note: if ct is in motif, center-type neighbors are already placed in first column as centers
        center_cells = np.unique(center_neighbor_pairs[:, 0])  # unique center cells
        neighbor_cells = np.unique(center_neighbor_pairs[:, 1])  # neighboring motif cells without center type

        # Get non-neighbor motif cells (set difference from all motif cells)
        motif_mask = np.isin(np.array(self.labels), motif)
        all_motif_cells = np.where(motif_mask)[0]
        non_neighbor_cells = np.setdiff1d(all_motif_cells, neighbor_cells)

        # Remove cells of center type from non-neighbor groups to focus on inter-cell-type interactions
        if ct_in_motif:
            center_cell_mask_non = self.labels[non_neighbor_cells] == ct
            non_neighbor_cells = non_neighbor_cells[~center_cell_mask_non]
            print(f'Focus on inter-cell-type interactions: Remove {center_cell_mask_non.sum()} center type cells from non-neighbor groups.')

        if len(non_neighbor_cells) < 10:
            raise ValueError(f"Not enough non-neighbor cells ({len(non_neighbor_cells)}) for correlation analysis. Need at least 5 cells.")

        print(f"Found {len(center_cells)} center cells with motif nearby.")
        print(f"Found {len(neighbor_cells)} motif neighbor cells.")
        print(f"Found {len(non_neighbor_cells)} non-neighbor motif cells.")
        print(f"Found {len(center_neighbor_pairs)} center-neighbor pairs.")
        

        # Get gene list
        if genes is None:
            genes = self.genes
        elif isinstance(genes, str):
            genes = [genes]

        # Filter genes that exist
        valid_genes = [g for g in genes if g in self.genes]
        if len(valid_genes) == 0:
            raise ValueError("No valid genes found in the dataset.")
        expr_genes = self.adata[:, valid_genes].X

        # Check if sparse
        is_sparse = sparse.issparse(expr_genes)

        # Filter genes by non-zero expression (work with sparse matrix)
        if is_sparse:
            nonzero_all = np.array((expr_genes > 0).sum(axis=0)).flatten()
        else:
            nonzero_all = (expr_genes > 0).sum(axis=0)

        valid_gene_mask = nonzero_all >= min_nonzero

        if valid_gene_mask.sum() == 0:
            raise ValueError(f"No genes passed the min_nonzero={min_nonzero} filter for cells of interest.")

        # Apply gene filter
        filtered_genes = [valid_genes[i] for i in range(len(valid_genes)) if valid_gene_mask[i]]
        expr_genes= expr_genes[:, valid_gene_mask]

        print(f"After filtering (min_nonzero={min_nonzero}): {len(filtered_genes)} genes")

        # Compute cell type means for ALL cell types (keep sparse)
        # We need cell-type-specific means for proper centering
        start_time = time()

        # Get all unique cell types in the dataset
        all_cell_types = np.unique(self.labels)
        cell_type_means = {}  # Dictionary to store mean expression for each cell type

        print("Computing cell-type-specific global means...")
        for cell_type in all_cell_types:
            ct_mask = self.labels == cell_type
            ct_cells = np.where(ct_mask)[0]
            if len(ct_cells) > 0:
                ct_expr = expr_genes[ct_cells, :]
                if is_sparse:
                    cell_type_means[cell_type] = np.array(ct_expr.mean(axis=0)).flatten()
                else:
                    cell_type_means[cell_type] = ct_expr.mean(axis=0)
        
        center_mean = cell_type_means[ct]


        # ==================================================================================
        # Correlation 1: Center with motif vs Neighboring motif (PAIRED DATA)
        # ==================================================================================
        print("\n" + "="*60)
        print("Computing Correlation 1: Center with motif vs Neighboring motif (paired)")
        print("Using cell-type-specific references for each pair")
        print("="*60)

        # Convert pairs to arrays
        pair_centers = center_neighbor_pairs[:, 0]
        pair_neighbors = center_neighbor_pairs[:, 1]

        # Extract paired expression data
        pair_center_expr = expr_genes[pair_centers, :]
        pair_neighbor_expr = expr_genes[pair_neighbors, :]

        # Get cell types for each neighbor in pairs
        neighbor_cell_types = self.labels[pair_neighbors]

        # Compute correlation keeping sparse matrices sparse
        n_genes = len(filtered_genes)
        n_pairs = len(pair_centers)

        if is_sparse:
            # Vectorized approach without for loops over cell types
            # Formula derivation:
            # Cov(X-μ_X, Y-μ^{ct_i}) = (1/n)Σ_i (x_i - μ_X)(y_i - μ^{ct_i})
            # = (1/n)Σ_i x_i*y_i - (1/n)Σ_i x_i*μ^{ct_i} - (μ_X/n)Σ_i y_i + (μ_X/n)Σ_i μ^{ct_i}
            # where: Σ_i x_i*μ^{ct_i} = Σ_ct [μ^ct * Σ_{i:ct_i=ct} x_i]
            #        Σ_i μ^{ct_i} = Σ_ct [μ^ct * |{i:ct_i=ct}|]

            # Get unique neighbor types and their counts directly
            unique_neighbor_types, type_counts = np.unique(neighbor_cell_types, return_counts=True)
            n_types = len(unique_neighbor_types)

            # Build mapping for creating indicator matrix
            type_to_idx = {ct: idx for idx, ct in enumerate(unique_neighbor_types)}
            type_indices = np.array([type_to_idx[ct] for ct in neighbor_cell_types])

            # Stack cell-type-specific means into a matrix (shape: n_genes x n_types)
            neighbor_type_means_matrix = np.column_stack([cell_type_means[ct] for ct in unique_neighbor_types])

            # ==================== Cross-covariance computation ====================
            # Note: Center cells are all of type 'ct', so using center_mean (global mean of ct) is correct

            # Term 1: (1/n) Σ_i x_i * y_i
            cross_product = pair_center_expr.T @ pair_neighbor_expr
            # Gene x Gene matrix is typically dense, convert once
            if sparse.issparse(cross_product):
                cross_product = np.asarray(cross_product.todense())
            term1 = cross_product / n_pairs

            # Term 2: -(1/n) Σ_i x_i * μ^{ct_i} = -(1/n) Σ_ct [μ^ct * Σ_{i:ct_i=ct} x_i]
            # Group center expression by neighbor type using sparse indexing
            # Create a sparse indicator matrix: (n_pairs x n_types), where [i, j] = 1 if pair i has neighbor type j
            from scipy.sparse import csr_matrix
            type_indicator = csr_matrix((np.ones(n_pairs), (np.arange(n_pairs), type_indices)),
                                       shape=(n_pairs, n_types))

            # Sum center expression grouped by neighbor type: (n_genes x n_types)
            sum_center_by_type = pair_center_expr.T @ type_indicator
            # Convert to dense for matrix multiplication with means (small matrix: n_genes x n_types)
            if sparse.issparse(sum_center_by_type):
                sum_center_by_type = np.asarray(sum_center_by_type.todense())

            # Compute: Σ_ct [μ^ct_g2 * Σ_{i:ct_i=ct} x_i_g1] for all gene pairs (g1, g2)
            # Result: (n_genes x n_genes)
            term2 = (sum_center_by_type @ neighbor_type_means_matrix.T) / n_pairs

            # Term 3: -(μ_X/n) Σ_i y_i
            # Keep sparse until final sum
            sum_neighbor = np.array(pair_neighbor_expr.sum(axis=0)).flatten()
            term3 = np.outer(center_mean, sum_neighbor / n_pairs)

            # Term 4: (μ_X/n) Σ_i μ^{ct_i} = (μ_X/n) Σ_ct [|ct| * μ^ct]
            weighted_neighbor_mean = (neighbor_type_means_matrix @ type_counts) / n_pairs
            term4 = np.outer(center_mean, weighted_neighbor_mean)

            # Final cross-covariance (all terms are now dense n_genes x n_genes)
            cross_cov_neighbor = term1 - term2 - term3 + term4

            # ==================== Variance computation ====================
            # Var(X - μ_X) = (1/n)Σ X² - 2μ_X*(1/n)ΣX + μ_X²
            # Note: Center cells are all type 'ct', so using center_mean is correct
            sum_sq_center = np.array(pair_center_expr.power(2).sum(axis=0)).flatten()
            sum_center = np.array(pair_center_expr.sum(axis=0)).flatten()
            var_center_paired = (sum_sq_center / n_pairs
                                - 2 * center_mean * sum_center / n_pairs
                                + center_mean**2)

            # Var(Y - μ^{ct_i}) = (1/n)Σ_i y_i² - (2/n)Σ_i y_i*μ^{ct_i} + (1/n)Σ_i (μ^{ct_i})²
            # Neighbor cells have heterogeneous types, use cell-type-specific means
            sum_sq_neighbor = np.array(pair_neighbor_expr.power(2).sum(axis=0)).flatten()

            # (1/n)Σ_i y_i*μ^{ct_i} = (1/n) Σ_ct [μ^ct * Σ_{i:ct_i=ct} y_i]
            sum_neighbor_by_type = pair_neighbor_expr.T @ type_indicator
            # Convert small matrix (n_genes x n_types) to dense
            if sparse.issparse(sum_neighbor_by_type):
                sum_neighbor_by_type = np.asarray(sum_neighbor_by_type.todense())
            # Element-wise multiply with type means and sum over types
            term_y_mean = (sum_neighbor_by_type * neighbor_type_means_matrix).sum(axis=1) / n_pairs

            # (1/n)Σ_i (μ^{ct_i})² = (1/n) Σ_ct [|ct| * (μ^ct)²]
            term_mean_sq = ((neighbor_type_means_matrix**2) @ type_counts) / n_pairs

            var_neighbor_paired = sum_sq_neighbor / n_pairs - 2 * term_y_mean + term_mean_sq

            std_center_paired = np.sqrt(np.maximum(var_center_paired, 0))
            std_neighbor_paired = np.sqrt(np.maximum(var_neighbor_paired, 0))
        else:
            # Dense matrix operations with cell-type-specific centering
            # Create a matrix of neighbor-type-specific means for each pair
            neighbor_type_means_matrix = np.array([cell_type_means[ct] for ct in neighbor_cell_types])

            pair_center_shifted = pair_center_expr - center_mean[np.newaxis, :]
            pair_neighbor_shifted = pair_neighbor_expr - neighbor_type_means_matrix

            # Compute covariance matrix
            cross_cov_neighbor = (pair_center_shifted.T @ pair_neighbor_shifted) / n_pairs

            # Compute standard deviations
            std_center_paired = np.sqrt(np.maximum((pair_center_shifted**2).sum(axis=0) / n_pairs, 0))
            std_neighbor_paired = np.sqrt(np.maximum((pair_neighbor_shifted**2).sum(axis=0) / n_pairs, 0))

        # Correlation matrix (common for both sparse and dense)
        std_outer_neighbor = np.outer(std_center_paired, std_neighbor_paired)
        std_outer_neighbor[std_outer_neighbor == 0] = 1e-10

        corr_matrix_neighbor = cross_cov_neighbor / std_outer_neighbor

        # Effective sample size for paired correlation
        center_unique = len(np.unique(pair_centers))
        neighbor_unique = len(np.unique(pair_neighbors))
        n_eff_neighbor = min(center_unique, neighbor_unique)

        print(f"Number of pairs: {n_pairs}")
        print(f"Unique center cells: {center_unique}")
        print(f"Unique neighbor cells: {neighbor_unique}")
        print(f"Neighbor cell types in pairs: {unique_neighbor_types if is_sparse else np.unique(neighbor_cell_types)}")
        print(f"Set effective sample size as minimum of unique center cells and neighbor cells: {n_eff_neighbor}")

        end_time = time()
        print(f"Time for computing correlation 1 matrix: {end_time - start_time:.4f} seconds")
        # ==================================================================================
        # Correlation 2: Center with motif vs Distant motif (ALL PAIRS)
        # ==================================================================================
        print("\n" + "="*60)
        print("Computing Correlation 2: Center with motif vs Distant motif (all pairs)")
        print("Using cell-type-specific references for non-neighbor cells")
        print("="*60)
        start_time = time()
        # Use the same center cells as correlation 1
        # Use distant motif cells (non-neighbor cells)
        center_expr_corr2 = expr_genes[center_cells, :]
        non_neighbor_expr = expr_genes[non_neighbor_cells, :]

        # Get cell types for non-neighbor cells
        non_neighbor_cell_types = self.labels[non_neighbor_cells]

        n_center = len(center_cells)
        n_non_neighbor = len(non_neighbor_cells)

        if is_sparse and sparse.issparse(center_expr_corr2):
            # Vectorized approach for all-to-all pairs without for loops
            # For all-to-all: each center cell paired with each non-neighbor cell
            # Cov(X-μ_X, Y-μ^{ct_j}) where j indexes non-neighbor cells
            # = (1/(n_c*n_nn)) Σ_i Σ_j x_i*y_j - (1/(n_c*n_nn)) Σ_i Σ_j x_i*μ^{ct_j}
            #   - (μ_X/(n_c*n_nn)) Σ_i Σ_j y_j + (μ_X/(n_c*n_nn)) Σ_i Σ_j μ^{ct_j}
            # Simplify: Σ_i Σ_j x_i*μ^{ct_j} = (Σ_i x_i) * (Σ_j μ^{ct_j}) = (Σ_i x_i) * Σ_ct [|ct|*μ^ct]

            # Get unique non-neighbor types and their counts directly
            unique_non_neighbor_types, type_counts = np.unique(non_neighbor_cell_types, return_counts=True)
            n_types = len(unique_non_neighbor_types)

            # Build mapping for creating indicator matrix
            type_to_idx = {ct: idx for idx, ct in enumerate(unique_non_neighbor_types)}
            type_indices = np.array([type_to_idx[ct] for ct in non_neighbor_cell_types])

            # Stack cell-type-specific means (n_genes x n_types)
            non_neighbor_type_means_matrix = np.column_stack([cell_type_means[ct] for ct in unique_non_neighbor_types])

            # Compute center statistics
            sum_center_corr2 = np.array(center_expr_corr2.sum(axis=0)).flatten()
            sum_sq_center_corr2 = np.array(center_expr_corr2.power(2).sum(axis=0)).flatten()

            # ==================== Cross-covariance computation ====================
            # Term 1: (1/(n_c*n_nn)) Σ_i Σ_j x_i*y_j = (1/(n_c*n_nn)) * (Σ_i x_i) * (Σ_j y_j)
            sum_non_neighbor = np.array(non_neighbor_expr.sum(axis=0)).flatten()
            term1 = np.outer(sum_center_corr2, sum_non_neighbor) / (n_center * n_non_neighbor)

            # Term 2: -(1/(n_c*n_nn)) Σ_i Σ_j x_i*μ^{ct_j} = -(1/(n_c*n_nn)) * (Σ_i x_i) * (Σ_ct |ct|*μ^ct)
            weighted_nn_mean = non_neighbor_type_means_matrix @ type_counts  # (n_genes,)
            term2 = np.outer(sum_center_corr2, weighted_nn_mean) / (n_center * n_non_neighbor)

            # Term 3: -(μ_X/(n_c*n_nn)) Σ_i Σ_j y_j = -(μ_X/(n_c*n_nn)) * n_c * (Σ_j y_j)
            term3 = np.outer(center_mean, sum_non_neighbor) / n_non_neighbor

            # Term 4: (μ_X/(n_c*n_nn)) Σ_i Σ_j μ^{ct_j} = (μ_X/(n_c*n_nn)) * n_c * (Σ_ct |ct|*μ^ct)
            term4 = np.outer(center_mean, weighted_nn_mean) / n_non_neighbor

            cross_cov_non_neighbor = term1 - term2 - term3 + term4

            # ==================== Variance computation ====================
            # Var(X - μ_X)
            var_center_corr2 = (sum_sq_center_corr2 / n_center
                        - 2 * center_mean * sum_center_corr2 / n_center
                        + center_mean**2)

            # Var(Y - μ^{ct_j}) = (1/n_nn) Σ_j y_j² - (2/n_nn) Σ_j y_j*μ^{ct_j} + (1/n_nn) Σ_j (μ^{ct_j})²
            sum_sq_non_neighbor = np.array(non_neighbor_expr.power(2).sum(axis=0)).flatten()

            # (1/n_nn) Σ_j y_j*μ^{ct_j} = (1/n_nn) Σ_ct [μ^ct * Σ_{j:ct_j=ct} y_j]
            from scipy.sparse import csr_matrix
            type_indicator = csr_matrix((np.ones(n_non_neighbor), (np.arange(n_non_neighbor), type_indices)),
                                       shape=(n_non_neighbor, n_types))
            sum_non_neighbor_by_type = non_neighbor_expr.T @ type_indicator
            # Convert small matrix (n_genes x n_types) to dense
            if sparse.issparse(sum_non_neighbor_by_type):
                sum_non_neighbor_by_type = np.asarray(sum_non_neighbor_by_type.todense())

            term_y_mean = (sum_non_neighbor_by_type * non_neighbor_type_means_matrix).sum(axis=1) / n_non_neighbor

            # (1/n_nn) Σ_j (μ^{ct_j})² = (1/n_nn) Σ_ct [|ct| * (μ^ct)²]
            term_mean_sq = ((non_neighbor_type_means_matrix**2) @ type_counts) / n_non_neighbor

            var_non_neighbor = sum_sq_non_neighbor / n_non_neighbor - 2 * term_y_mean + term_mean_sq

            std_center_corr2 = np.sqrt(np.maximum(var_center_corr2, 0))
            std_non_neighbor = np.sqrt(np.maximum(var_non_neighbor, 0))
        else:
            # Dense matrix operations with cell-type-specific centering
            # Create a matrix of cell-type-specific means for each non-neighbor cell
            non_neighbor_type_means_matrix = np.array([cell_type_means[ct] for ct in non_neighbor_cell_types])

            # Center the data
            center_expr_shifted = center_expr_corr2 - center_mean[np.newaxis, :]
            non_neighbor_expr_shifted = non_neighbor_expr - non_neighbor_type_means_matrix

            # For all-to-all pairs: sum over all combinations
            # Total cross-product = (sum of shifted center) × (sum of shifted non-neighbor)
            sum_center_shifted = center_expr_shifted.sum(axis=0)
            sum_non_neighbor_shifted = non_neighbor_expr_shifted.sum(axis=0)

            cross_cov_non_neighbor = np.outer(sum_center_shifted, sum_non_neighbor_shifted) / (n_center * n_non_neighbor)

            # Variances
            var_center_corr2 = (center_expr_shifted**2).sum(axis=0) / n_center
            var_non_neighbor = (non_neighbor_expr_shifted**2).sum(axis=0) / n_non_neighbor

            std_center_corr2 = np.sqrt(np.maximum(var_center_corr2, 0))
            std_non_neighbor = np.sqrt(np.maximum(var_non_neighbor, 0))

        std_outer_non_neighbor = np.outer(std_center_corr2, std_non_neighbor)
        std_outer_non_neighbor[std_outer_non_neighbor == 0] = 1e-10

        corr_matrix_non_neighbor = cross_cov_non_neighbor / std_outer_non_neighbor

        # Effective sample size
        n_eff_non_neighbor = min(n_center, n_non_neighbor)

        print(f"Center cells: {n_center}")
        print(f"Distant motif cells: {n_non_neighbor}")
        print(f"Distant motif cell types: {unique_non_neighbor_types if is_sparse else np.unique(non_neighbor_cell_types)}")
        print(f"Effective sample size: {n_eff_non_neighbor}")

        end_time = time()
        print(f'\nTime for computing correlations 2: {end_time-start_time:.2f} seconds')

        # ==================================================================================
        # Correlation 3: Center without motif vs Neighbors (PAIRED DATA)
        # ==================================================================================
        print("\n" + "="*60)
        print("Computing Correlation 3: Center without motif vs Neighbors (paired)")
        print("="*60)

        # Get neighbors for centers WITHOUT motif by excluding centers with motif
        start_time = time()
        no_motif_result = self.get_all_neighbor_cells(
            ct=ct,
            max_dist=max_dist,
            k=k,
            min_size=min_size,
            exclude_centers=center_cells,
            exclude_neighbors=neighbor_cells,
        )

        center_no_motif_pairs = no_motif_result['center_neighbor_pairs']

        if len(center_no_motif_pairs) < 10:
            print(f"Not enough pairs ({len(center_no_motif_pairs)}) for centers without motif. Skipping.")
            corr_matrix_no_motif = None
            n_eff_no_motif = 0
            centers_without_motif = np.array([])
            centers_without_motif_neighbors = np.array([])
        else:
            # Extract unique center and neighbor cells from pairs
            centers_without_motif = np.unique(center_no_motif_pairs[:, 0])
            centers_without_motif_neighbors = np.unique(center_no_motif_pairs[:, 1])

            print(f"Found {len(centers_without_motif)} center cells without motif nearby")
            print(f"Found {len(centers_without_motif_neighbors)} neighbor cells for centers without motif")
            print(f"Found {len(center_no_motif_pairs)} center-neighbor pairs")

            # Extract paired data
            pair_centers_no_motif = center_no_motif_pairs[:, 0]
            pair_neighbors_no_motif = center_no_motif_pairs[:, 1]

            # Extract paired expression data
            pair_center_no_motif_expr = expr_genes[pair_centers_no_motif, :]
            pair_neighbor_no_motif_expr = expr_genes[pair_neighbors_no_motif, :]

            # Get cell types for each neighbor in pairs
            neighbor_no_motif_cell_types = self.labels[pair_neighbors_no_motif]

            # Compute correlation keeping sparse matrices sparse
            n_pairs_no_motif = len(pair_centers_no_motif)

            if is_sparse and sparse.issparse(pair_center_no_motif_expr):
                # Vectorized approach without for loops - same as Correlation 1
                # Get unique neighbor types and their counts directly
                unique_neighbor_no_motif_types, type_counts = np.unique(neighbor_no_motif_cell_types, return_counts=True)
                n_types = len(unique_neighbor_no_motif_types)

                # Build mapping for creating indicator matrix
                type_to_idx = {ct: idx for idx, ct in enumerate(unique_neighbor_no_motif_types)}
                type_indices = np.array([type_to_idx[ct] for ct in neighbor_no_motif_cell_types])

                # Stack cell-type-specific means (n_genes x n_types)
                neighbor_type_means_matrix = np.column_stack([cell_type_means[ct] for ct in unique_neighbor_no_motif_types])

                # ==================== Cross-covariance computation ====================
                # Note: Center cells are all type 'ct', so using center_mean is correct

                # Term 1: (1/n) Σ_i x_i * y_i
                cross_product = pair_center_no_motif_expr.T @ pair_neighbor_no_motif_expr
                # Gene x Gene matrix is typically dense, convert once
                if sparse.issparse(cross_product):
                    cross_product = np.asarray(cross_product.todense())
                term1 = cross_product / n_pairs_no_motif

                # Term 2: -(1/n) Σ_i x_i * μ^{ct_i} = -(1/n) Σ_ct [μ^ct * Σ_{i:ct_i=ct} x_i]
                from scipy.sparse import csr_matrix
                type_indicator = csr_matrix((np.ones(n_pairs_no_motif), (np.arange(n_pairs_no_motif), type_indices)),
                                           shape=(n_pairs_no_motif, n_types))

                sum_center_by_type = pair_center_no_motif_expr.T @ type_indicator
                # Convert small matrix (n_genes x n_types) to dense
                if sparse.issparse(sum_center_by_type):
                    sum_center_by_type = np.asarray(sum_center_by_type.todense())

                term2 = (sum_center_by_type @ neighbor_type_means_matrix.T) / n_pairs_no_motif

                # Term 3: -(μ_X/n) Σ_i y_i
                sum_neighbor = np.array(pair_neighbor_no_motif_expr.sum(axis=0)).flatten()
                term3 = np.outer(center_mean, sum_neighbor / n_pairs_no_motif)

                # Term 4: (μ_X/n) Σ_i μ^{ct_i} = (μ_X/n) Σ_ct [|ct| * μ^ct]
                weighted_neighbor_mean = (neighbor_type_means_matrix @ type_counts) / n_pairs_no_motif
                term4 = np.outer(center_mean, weighted_neighbor_mean)

                cross_cov_no_motif = term1 - term2 - term3 + term4

                # ==================== Variance computation ====================
                # Var(X - μ_X): Center cells are all type 'ct'
                sum_sq_center = np.array(pair_center_no_motif_expr.power(2).sum(axis=0)).flatten()
                sum_center = np.array(pair_center_no_motif_expr.sum(axis=0)).flatten()
                var_center_no_motif = (sum_sq_center / n_pairs_no_motif
                                    - 2 * center_mean * sum_center / n_pairs_no_motif
                                    + center_mean**2)

                # Var(Y - μ^{ct_i}): Neighbor cells have heterogeneous types
                sum_sq_neighbor = np.array(pair_neighbor_no_motif_expr.power(2).sum(axis=0)).flatten()

                sum_neighbor_by_type = pair_neighbor_no_motif_expr.T @ type_indicator
                # Convert small matrix (n_genes x n_types) to dense
                if sparse.issparse(sum_neighbor_by_type):
                    sum_neighbor_by_type = np.asarray(sum_neighbor_by_type.todense())

                term_y_mean = (sum_neighbor_by_type * neighbor_type_means_matrix).sum(axis=1) / n_pairs_no_motif
                term_mean_sq = ((neighbor_type_means_matrix**2) @ type_counts) / n_pairs_no_motif

                var_neighbor_no_motif = sum_sq_neighbor / n_pairs_no_motif - 2 * term_y_mean + term_mean_sq

                std_center_no_motif = np.sqrt(np.maximum(var_center_no_motif, 0))
                std_neighbor_no_motif = np.sqrt(np.maximum(var_neighbor_no_motif, 0))
            else:
                # Dense matrix operations with cell-type-specific centering
                # Create a matrix of neighbor-type-specific means for each pair
                neighbor_type_means_matrix = np.array([cell_type_means[ct] for ct in neighbor_no_motif_cell_types])

                pair_center_no_motif_shifted = pair_center_no_motif_expr - center_mean[np.newaxis, :]
                pair_neighbor_no_motif_shifted = pair_neighbor_no_motif_expr - neighbor_type_means_matrix

                # Compute correlation
                cross_cov_no_motif = (pair_center_no_motif_shifted.T @ pair_neighbor_no_motif_shifted) / n_pairs_no_motif

                # Standard deviations
                std_center_no_motif = np.sqrt(np.maximum((pair_center_no_motif_shifted**2).sum(axis=0) / n_pairs_no_motif, 0))
                std_neighbor_no_motif = np.sqrt(np.maximum((pair_neighbor_no_motif_shifted**2).sum(axis=0) / n_pairs_no_motif, 0))

            std_outer_no_motif = np.outer(std_center_no_motif, std_neighbor_no_motif)
            std_outer_no_motif[std_outer_no_motif == 0] = 1e-10

            corr_matrix_no_motif = cross_cov_no_motif / std_outer_no_motif

            # Effective sample size
            center_no_motif_unique = len(np.unique(pair_centers_no_motif))
            neighbor_no_motif_unique = len(np.unique(pair_neighbors_no_motif))
            n_eff_no_motif = min(center_no_motif_unique, neighbor_no_motif_unique)

            print(f"Unique center cells: {center_no_motif_unique}")
            print(f"Unique neighbor cells: {neighbor_no_motif_unique}")
            print(f"Neighbor cell types in pairs: {unique_neighbor_no_motif_types if is_sparse else np.unique(neighbor_no_motif_cell_types)}")
            print(f"Effective sample size: {n_eff_no_motif}")

        end_time = time()
        print(f"time of computing correlation-3 matrix is {end_time - start_time:.2f} seconds.")

        # Prepare results - combine all three correlations and perform statistical tests
        print("\n" + "="*60)
        print("Performing Fisher Z-tests and preparing results")
        print("="*60)

        start_time = time()

        # Fisher Z transformation (vectorized)
        def fisher_z_transform(r):
            """Fisher Z transformation for correlation coefficient"""
            r_clipped = np.clip(r, -0.9999, 0.9999)
            return 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))

        def fisher_z_test_vectorized(r1, n1, r2, n2):
            """
            Vectorized Fisher Z-test for comparing correlation matrices.
            Two-tailed test: H0: r1 = r2, H1: r1 != r2

            Parameters:
            r1, r2: correlation matrices (can be 2D arrays)
            n1, n2: effective sample sizes (scalars)

            Returns:
            z_stat: Z statistic matrix
            p_value: two-tailed p-value matrix
            """
            z1 = fisher_z_transform(r1)
            z2 = fisher_z_transform(r2)

            se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))
            z_stat = (z1 - z2) / se_diff

            # Two-tailed p-value: P(|Z| > |z_stat|)
            # This detects significant differences in both directions
            p_value = 2 * (1 - scipy_stats.norm.cdf(np.abs(z_stat)))

            return z_stat, p_value

        # Test 1: Corr1 vs Corr2 (neighbor vs non-neighbor) - vectorized
        _, p_value_test1 = fisher_z_test_vectorized(
            corr_matrix_neighbor, n_eff_neighbor,
            corr_matrix_non_neighbor, n_eff_non_neighbor
        )
        delta_corr_test1 = corr_matrix_neighbor - corr_matrix_non_neighbor

        # Test 2: Corr1 vs Corr3 (neighbor vs no_motif) - vectorized if available
        if corr_matrix_no_motif is not None:
            _, p_value_test2 = fisher_z_test_vectorized(
                corr_matrix_neighbor, n_eff_neighbor,
                corr_matrix_no_motif, n_eff_no_motif
            )
            delta_corr_test2 = corr_matrix_neighbor - corr_matrix_no_motif

            # Combined score (vectorized)
            combined_score = (0.3 * delta_corr_test1 * (-np.log10(p_value_test1 + 1e-300)) +
                            0.7 * delta_corr_test2 * (-np.log10(p_value_test2 + 1e-300)))
        else:
            p_value_test2 = None
            delta_corr_test2 = None
            combined_score = None

        # Create meshgrid for gene pairs (vectorized)
        gene_center_idx, gene_motif_idx = np.meshgrid(np.arange(n_genes), np.arange(n_genes), indexing='ij')

        # Build results DataFrame (vectorized)
        results_df = pd.DataFrame({
            'gene_center': np.array(filtered_genes)[gene_center_idx.flatten()],
            'gene_motif': np.array(filtered_genes)[gene_motif_idx.flatten()],
            'corr_neighbor': corr_matrix_neighbor.flatten(),
            'corr_non_neighbor': corr_matrix_non_neighbor.flatten(),
            'corr_diff_neighbor_vs_non': delta_corr_test1.flatten(),
            'p_value_test1': p_value_test1.flatten(),
            'delta_corr_test1': delta_corr_test1.flatten(),
        })

        # Add test2 results if available
        if corr_matrix_no_motif is not None:
            results_df['corr_center_no_motif'] = corr_matrix_no_motif.flatten()
            results_df['p_value_test2'] = p_value_test2.flatten()
            results_df['delta_corr_test2'] = delta_corr_test2.flatten()
            results_df['combined_score'] = combined_score.flatten()
        else:
            print("Note: Test2 not available")
            results_df['corr_center_no_motif'] = np.nan
            results_df['p_value_test2'] = np.nan
            results_df['delta_corr_test2'] = np.nan
            results_df['combined_score'] = np.nan

        # Apply FDR correction accounting for ALL tests performed
        # Strategy: Pool all p-values from both tests, apply FDR jointly
        print(f"\nTotal gene pairs: {len(results_df)}")

        if corr_matrix_no_motif is not None:
            # Step 1: Filter by direction consistency first
            # Both delta_corr should have the same sign (both positive or both negative)
            same_direction = np.sign(results_df['delta_corr_test1']) == np.sign(results_df['delta_corr_test2'])
            print(f"Gene pairs with consistent covarying direction: {same_direction.sum()}")

            if same_direction.sum() > 0:
                # Step 2: Pool all p-values from both tests for FDR correction
                # This accounts for the fact that we perform 2 × n_gene_pairs tests
                p_values_test1 = results_df.loc[same_direction, 'p_value_test1'].values
                p_values_test2 = results_df.loc[same_direction, 'p_value_test2'].values

                assert len(p_values_test1) == len(p_values_test2), "Inconsistent number of p-values!"
                # Concatenate all p-values: total = 2 × n_gene_pairs
                all_p_values = np.concatenate([p_values_test1, p_values_test2])
                n_consistent = same_direction.sum()

                print(f"Applying FDR correction to {len(all_p_values)} total tests (2 × {n_consistent} gene pairs)")

                # Apply FDR correction to ALL pooled p-values
                reject_all, q_values_all, _, _ = multipletests(
                    all_p_values,
                    alpha=0.05,
                    method='fdr_bh'
                )

                # Split results back to test1 and test2
                q_values_test1 = q_values_all[:n_consistent]
                q_values_test2 = q_values_all[n_consistent:]
                reject_test1 = reject_all[:n_consistent]
                reject_test2 = reject_all[n_consistent:]

                # Assign q-values and rejection status
                results_df.loc[same_direction, 'q_value_test1'] = q_values_test1
                results_df.loc[same_direction, 'q_value_test2'] = q_values_test2
                results_df.loc[same_direction, 'reject_test1_fdr'] = reject_test1
                results_df.loc[same_direction, 'reject_test2_fdr'] = reject_test2

                # For inconsistent direction, set q-values to NaN
                # 只保留same_direction的pair
                results_df = results_df[same_direction]

                # Count gene pairs passing both FDR thresholds
                mask_both_fdr = reject_test1 & reject_test2
                n_both_fdr = mask_both_fdr.sum()

                print(f"\nFDR correction results (joint across both tests):")
                print(f"  - Test1 FDR significant (q < 0.05): {reject_test1.sum()}")
                print(f"  - Test2 FDR significant (q < 0.05): {reject_test2.sum()}")
                print(f"  - Both tests FDR significant: {n_both_fdr}")
            else:
                results_df['q_value_test1'] = np.nan
                results_df['q_value_test2'] = np.nan
                results_df['reject_test1_fdr'] = False
                results_df['reject_test2_fdr'] = False
                print("No gene pairs with consistent direction found.")
        else:
            # Only test1 available - apply FDR to all test1 p-values
            print("Note: Test2 not available (no centers without motif)")
            p_values_test1_all = results_df['p_value_test1'].values
            reject_test1, q_values_test1, _, _ = multipletests(
                p_values_test1_all,
                alpha=0.05,
                method='fdr_bh'
            )
            results_df['q_value_test1'] = q_values_test1
            results_df['reject_test1_fdr'] = reject_test1
            results_df['q_value_test2'] = np.nan
            results_df['reject_test2_fdr'] = False

            print(f"FDR correction applied to all {len(results_df)} gene pairs:")
            print(f"  - Test1 FDR significant (q < 0.05): {reject_test1.sum()}")

        # Sort by absolute value of combined score (descending) to capture both positive and negative co-varying
        if corr_matrix_no_motif is not None:
            results_df['abs_combined_score'] = np.abs(results_df['combined_score'])
            results_df = results_df.sort_values('abs_combined_score', ascending=False, na_position='last').reset_index(drop=True)
        else:
            # Sort by absolute value of test1 score if test2 not available
            results_df['test1_score'] = results_df['delta_corr_test1'] * (-np.log10(results_df['p_value_test1'] + 1e-300))
            results_df['abs_test1_score'] = np.abs(results_df['test1_score'])
            results_df = results_df.sort_values('abs_test1_score', ascending=False, na_position='last')

        print(f"\nResults prepared and sorted by {'absolute value of combined_score' if corr_matrix_no_motif is not None else 'absolute value of test1_score'}")
        end_time = time()
        print(f"Time for testing results: {end_time - start_time:.2f} seconds")

        # Prepare cell groups dictionary with cell pairs for each correlation
        cell_groups = {
            'center_neighbor_motif_pair': center_neighbor_pairs,  # Shape: (n_pairs, 2), columns: [center_idx, neighbor_idx]
            'non-neighbor_motif_cells': non_neighbor_cells,
            'non_motif_center_neighbor_pair': center_no_motif_pairs if corr_matrix_no_motif is not None else np.array([]).reshape(0, 2),
        }

        return results_df, cell_groups

    def plot_motif_celltype(self,
                            ct: str,
                            motif: Union[str, List[str]],
                            max_dist: float = 100,
                            fig_size: tuple = (5, 5),
                            save_path: Optional[str] = None
                            ):
        """
        Display the distribution of interested motifs in the radius-based neighborhood of certain cell type.
        This function is mainly used to visualize the results of motif_enrichment_dist. Make sure the input parameters
        are consistent with those of motif_enrichment_dist.

        Parameter
        ---------
        ct:
            Cell type as the center cells.
        motif:
            Motif (names of cell types) to be colored.
        max_dist:
            Spacing distance for building grid. Make sure using the same value as that in find_patterns_grid.
        fig_size:
            Figure size.
        save_path:
            Path to save the figure.
            If None, the figure will not be saved.

        Return
        ------
        A figure.
        """
        if isinstance(motif, str):
            motif = [motif]

        max_dist = min(max_dist, self.max_radius)

        motif_exc = [m for m in motif if m not in self.labels.unique()]
        if len(motif_exc) != 0:
            print(f"Found no {motif_exc} in {self.label_key}. Ignoring them.")
        motif = [m for m in motif if m not in motif_exc]

        if ct not in self.labels.unique():
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        cinds = [i for i, label in enumerate(self.labels) if label == ct]  # id of center cell type
        # ct_pos = self.spatial_pos[cinds]
        idxs = self.kd_tree.query_ball_point(self.spatial_pos, r=max_dist, return_sorted=False, workers=-1)

        # find the index of cell type spots whose neighborhoods contain given motif
        # cind_with_motif = []
        # sort_motif = sorted(motif)
        label_encoder = LabelEncoder()
        int_labels = label_encoder.fit_transform(np.array(self.labels))
        int_ct = label_encoder.transform(np.array(ct, dtype=object, ndmin=1))
        int_motifs = label_encoder.transform(np.array(motif))

        num_cells = len(idxs)
        num_types = len(label_encoder.classes_)
        idxs_filter = [np.array(ids)[np.array(ids) != i] for i, ids in enumerate(idxs)]

        flat_neighbors = np.concatenate(idxs_filter)
        row_indices = np.repeat(np.arange(num_cells), [len(neigh) for neigh in idxs_filter])
        neighbor_labels = int_labels[flat_neighbors]

        neighbor_matrix = np.zeros((num_cells, num_types), dtype=int)
        np.add.at(neighbor_matrix, (row_indices, neighbor_labels), 1)

        mask = int_labels == int_ct
        inds = np.where(np.all(neighbor_matrix[mask][:, int_motifs] > 0, axis=1))[0]
        cind_with_motif = [cinds[i] for i in inds]

        # for id in cinds:
        #
        #     if self.has_motif(sort_motif, [self.labels[idx] for idx in idxs[id] if idx != id]):
        #         cind_with_motif.append(id)

        # Locate the index of motifs in the neighborhood of center cell type.

        motif_mask = np.isin(np.array(self.labels), motif)
        all_neighbors = np.concatenate(idxs[cind_with_motif])
        exclude_self_mask = ~np.isin(all_neighbors, cind_with_motif)
        valid_neighbors = all_neighbors[motif_mask[all_neighbors] & exclude_self_mask]
        id_motif_celltype = set(valid_neighbors)

        # id_motif_celltype = set()
        # for id in cind_with_motif:
        #     id_neighbor = [i for i in idxs[id] if self.labels[i] in motif and i != id]
        #     id_motif_celltype.update(id_neighbor)

        # Plot figures
        motif_unique = set(motif)
        n_colors = len(motif_unique)
        colors = sns.color_palette('hsv', n_colors)
        color_map = {ct: col for ct, col in zip(motif_unique, colors)}
        motif_spot_pos = self.spatial_pos[list(id_motif_celltype), :]
        motif_spot_label = self.labels[list(id_motif_celltype)]
        fig, ax = plt.subplots(figsize=fig_size)
        # Plotting other spots as background
        labels_length = len(self.labels)
        id_motif_celltype_set = set(id_motif_celltype)
        cind_with_motif_set = set(cind_with_motif)
        bg_index = [i for i in range(labels_length) if i not in id_motif_celltype_set and i not in cind_with_motif_set]
        bg_adata = self.spatial_pos[bg_index, :]
        ax.scatter(bg_adata[:, 0],
                   bg_adata[:, 1],
                   color='darkgrey', s=1)
        # Plot center the cell type whose neighborhood contains motif
        ax.scatter(self.spatial_pos[cind_with_motif, 0],
                   self.spatial_pos[cind_with_motif, 1],
                   label=ct, edgecolors='red', facecolors='none', s=3,
                   )
        for ct_m in motif_unique:
            ct_ind = motif_spot_label == ct_m
            ax.scatter(motif_spot_pos[ct_ind, 0],
                       motif_spot_pos[ct_ind, 1],
                       label=ct_m, color=color_map[ct_m], s=1)

        ax.legend(title='motif', loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)
        # ax.legend(title='motif', loc='lower center', bbox_to_anchor=(1, 0.5), markerscale=4)
        plt.xlabel('Spatial X')
        plt.ylabel('Spatial Y')
        plt.title(f"Spatial distribution of motif around {ct}")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        # plt.tight_layout(rect=[0, 0, 1.1, 1])
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_all_center_motif(
            self,
            center: str,
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

        # Set labels for each group
        adata = self.adata.copy()
        adata.obs['tmp'] = 'other'

        center_with_motif = np.unique(ids['center_neighbor_motif_pair'][:, 0])
        center_without_motif = np.unique(ids['non_motif_center_neighbor_pair'][:, 0])
        neighbor_motif = np.unique(ids['center_neighbor_motif_pair'][:, 1]) 
        non_neighbor_motif = np.unique(ids['non-neighbor_motif_cells'])
        center_without_motif_neighbors = np.unique(ids['non_motif_center_neighbor_pair'][:, 1])


        adata.obs.iloc[center_with_motif, adata.obs.columns.get_loc('tmp')] = f'center {center} with motif'
        adata.obs.iloc[center_without_motif, adata.obs.columns.get_loc('tmp')] = f'non-motif center {center}'

        adata.obs.iloc[neighbor_motif, adata.obs.columns.get_loc('tmp')] = [f'neighbor motif: {self.labels[i]}' for i in neighbor_motif]
        adata.obs.iloc[non_neighbor_motif, adata.obs.columns.get_loc('tmp')] = [f'non-neighbor motif: {self.labels[i]}' for i in non_neighbor_motif]

        adata.obs.iloc[center_without_motif_neighbors, adata.obs.columns.get_loc('tmp')] = 'non-motif-center neighbors'

        neighbor_motif_types = adata.obs.iloc[neighbor_motif, adata.obs.columns.get_loc(self.label_key)].unique()
        non_neighbor_motif_types = adata.obs.iloc[non_neighbor_motif, adata.obs.columns.get_loc(self.label_key)].unique()

        color_dict = {
            'other': '#D3D3D3',  # light gray for other cells
            f'center {center} with motif': "#9F0707",  # dark red 
            f'non-motif center {center}': "#075FB1",  # dark blue
            'non-motif-center neighbors': "#6DE7E9"  # light blue
        }

        # Set red colors for neighbor motif and purple colors for non-neighbor motif
        n_neighbor_types = len(neighbor_motif_types)
        if n_neighbor_types > 0:
            red_colors = cm.Reds(np.linspace(0.3, 0.6, n_neighbor_types))
            for i, cell_type in enumerate(neighbor_motif_types):
                color_dict[f'neighbor motif: {cell_type}'] = red_colors[i]

        # Set purple colors for non-neighbor motif
        n_non_neighbor_types = len(non_neighbor_motif_types)
        if n_non_neighbor_types > 0:
            purple_colors = cm.Greens(np.linspace(0.4, 0.8, n_non_neighbor_types))
            for i, cell_type in enumerate(non_neighbor_motif_types):
                color_dict[f'non-neighbor motif: {cell_type}'] = purple_colors[i]

        # Plot figure
        fig, ax = plt.subplots(figsize=figsize)
        sc.pl.embedding(adata, basis='X_spatial', color='tmp', palette=color_dict, size=10, ax=ax, show=False)
        ax.set_title(f'Cell types around {center} with motif', fontsize=10)
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
