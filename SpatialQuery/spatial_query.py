
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
from scipy.sparse import csr_matrix
from statsmodels.stats.multitest import multipletests

from time import time

from .scfind4sp import SCFind
import scanpy as sc
from . import spatial_utils
from . import plotting



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
                           figsize: tuple = (10, 5),
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
        figsize:
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
                           max_dist: float = 100,
                           n_points: int = 1000,
                           min_support: float = 0.5,
                           min_size: int = 0,
                           if_display: bool = True,
                           figsize: tuple = (10, 5),
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
        figsize:
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
        if self.adata is None:
            raise ValueError("Expression data (adata) is not available. Please set build_gene_index=False when initializing spatial_query.")
        
        motif = motif if isinstance(motif, list) else [motif]

        # Get neighbor and non-neighbor cell IDs plus paired mappings (using original motif)
        neighbor_result = spatial_utils.get_motif_neighbor_cells(sq_obj=self, ct=ct, motif=motif, max_dist=max_dist, k=k, min_size=min_size)

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
            print(f'Focus on inter-cell-type interactions: Remove center type cells from non-neighbor groups.')

        if len(non_neighbor_cells) < 10:
            raise ValueError(f"Not enough non-neighbor cells ({len(non_neighbor_cells)}) for correlation analysis. Need at least 5 cells.")
        

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
        print("="*60)

        # Convert pairs to arrays
        pair_centers = center_neighbor_pairs[:, 0]
        pair_neighbors = center_neighbor_pairs[:, 1]

        # Get cell types for each neighbor in pairs
        neighbor_cell_types = self.labels[pair_neighbors]

        # Compute correlation using helper function
        n_genes = len(filtered_genes)
        n_pairs = len(pair_centers)

        corr_matrix_neighbor, n_eff_neighbor = spatial_utils.compute_cross_correlation_paired(
            sq_obj=self,
            expr_genes=expr_genes,
            pair_centers=pair_centers,
            pair_neighbors=pair_neighbors,
            center_mean=center_mean,
            cell_type_means=cell_type_means,
            neighbor_cell_types=neighbor_cell_types,
            is_sparse=is_sparse
        )

        print(f"Number of pairs: {n_pairs}")
        print(f"Unique center cells: {len(np.unique(pair_centers))}")
        print(f"Unique neighbor cells: {len(np.unique(pair_neighbors))}")
        print(f"Effective sample size: {n_eff_neighbor}")

        end_time = time()
        print(f"Time for computing correlation-1 matrix: {end_time - start_time:.4f} seconds")
        # ==================================================================================
        # Correlation 2: Center with motif vs Distant motif (ALL PAIRS)
        # ==================================================================================
        print("\n" + "="*60)
        print("Computing Correlation 2: Center with motif vs Distant motif (all pairs)")
        print("="*60)
        start_time = time()

        # Get cell types for non-neighbor cells
        non_neighbor_cell_types = self.labels[non_neighbor_cells]

        n_center = len(center_cells)
        n_non_neighbor = len(non_neighbor_cells)

        corr_matrix_non_neighbor, n_eff_non_neighbor = spatial_utils.compute_cross_correlation_all_to_all(
            sq_obj=self,
            expr_genes=expr_genes,
            center_cells=center_cells,
            non_neighbor_cells=non_neighbor_cells,
            center_mean=center_mean,
            cell_type_means=cell_type_means,
            non_neighbor_cell_types=non_neighbor_cell_types,
            is_sparse=is_sparse
        )

        print(f"Unique center cells: {n_center}")
        print(f"Unique distant cells: {n_non_neighbor}")
        print(f"Effective sample size: {n_eff_non_neighbor}")

        end_time = time()
        print(f'Time for computing correlations-2 matrix: {end_time-start_time:.2f} seconds')

        # ==================================================================================
        # Correlation 3: Center without motif vs Neighbors (PAIRED DATA)
        # ==================================================================================
        print("\n" + "="*60)
        print("Computing Correlation 3: Center without motif vs Neighbors (paired)")
        print("="*60)

        # Get neighbors for centers WITHOUT motif by excluding centers with motif
        start_time = time()
        no_motif_result = spatial_utils.get_all_neighbor_cells(sq_obj=self,
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
        else:
            # Extract paired data
            pair_centers_no_motif = center_no_motif_pairs[:, 0]
            pair_neighbors_no_motif = center_no_motif_pairs[:, 1]

            # Get cell types for each neighbor in pairs
            neighbor_no_motif_cell_types = self.labels[pair_neighbors_no_motif]

            corr_matrix_no_motif, n_eff_no_motif = spatial_utils.compute_cross_correlation_paired(
                sq_obj=self,
                expr_genes=expr_genes,
                pair_centers=pair_centers_no_motif,
                pair_neighbors=pair_neighbors_no_motif,
                center_mean=center_mean,
                cell_type_means=cell_type_means,
                neighbor_cell_types=neighbor_no_motif_cell_types,
                is_sparse=is_sparse
            )
            
            print(f"Number of pairs: {len(center_no_motif_pairs)}")
            print(f"Unique center cells: {len(np.unique(pair_centers_no_motif))}")
            print(f"Unique neighbor cells: {len(np.unique(pair_neighbors_no_motif))}")
            print(f"Effective sample size: {n_eff_no_motif}")

        end_time = time()
        print(f"time of computing correlation-3 matrix is {end_time - start_time:.2f} seconds.")

        # Prepare results - combine all three correlations and perform statistical tests
        print("\n" + "="*60)
        print("Performing Fisher Z-tests and preparing results")
        print("="*60)

        start_time = time()

        # Test 1: Corr1 vs Corr2 (neighbor vs non-neighbor) - vectorized
        _, p_value_test1 = spatial_utils.fisher_z_test(
            corr_matrix_neighbor, n_eff_neighbor,
            corr_matrix_non_neighbor, n_eff_non_neighbor
        )
        delta_corr_test1 = corr_matrix_neighbor - corr_matrix_non_neighbor

        # Test 2: Corr1 vs Corr3 (neighbor vs no_motif) - vectorized if available
        if corr_matrix_no_motif is not None:
            _, p_value_test2 = spatial_utils.fisher_z_test(
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
        print(f"Total gene pairs: {len(results_df)}")

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
            results_df['if_significant'] = results_df['reject_test1_fdr'] & results_df['reject_test2_fdr']
        else:
            # Sort by absolute value of test1 score if test2 not available
            results_df['test1_score'] = results_df['delta_corr_test1'] * (-np.log10(results_df['p_value_test1'] + 1e-300))
            results_df['combined_score'] = results_df['test1_score']
            results_df['abs_combined_score'] = np.abs(results_df['combined_score'])
            results_df = results_df.sort_values('abs_combined_score', ascending=False, na_position='last')
            results_df['if_significant'] = results_df['reject_test1_fdr'] & results_df['reject_test2_fdr']


        end_time = time()
        print(f"Time for testing results: {end_time - start_time:.2f} seconds\n")

        # Prepare cell groups dictionary with cell pairs for each correlation
        cell_groups = {
            'center_neighbor_motif_pair': center_neighbor_pairs,  # Shape: (n_pairs, 2), columns: [center_idx, neighbor_idx]
            'non-neighbor_motif_cells': non_neighbor_cells,
            'non_motif_center_neighbor_pair': center_no_motif_pairs if corr_matrix_no_motif is not None else np.array([]).reshape(0, 2),
        }

        return results_df, cell_groups

    def compute_gene_gene_correlation_by_type(self,
                                             ct: str,
                                             motif: Union[str, List[str]],
                                             genes: Optional[Union[str, List[str]]] = None,
                                             max_dist: Optional[float] = None,
                                             k: Optional[int] = None,
                                             min_size: int = 0,
                                             min_nonzero: int = 10,
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
        if self.adata is None:
            raise ValueError("Expression data (adata) is not available. Please set build_gene_index=False when initializing spatial_query.")

        motif = motif if isinstance(motif, list) else [motif]

        # Get non-center cell types in motif
        non_center_types = [m for m in motif if m != ct]

        if len(non_center_types) == 1:
            print(f"Only one non-center cell type in motif: {non_center_types}. Using compute_gene_gene_correlation method.")
            result, _ = self.compute_gene_gene_correlation(
                ct=ct,
                motif=motif,
                genes=genes,
                max_dist=max_dist,
                k=k,
                min_size=min_size,
                min_nonzero=min_nonzero
            )
            return result
        elif len(non_center_types) == 0:
            raise ValueError("Error: Only center cell type in motif. Please ensure motif includes at least one non-center cell type.")

        print(f"Analyzing {len(non_center_types)} non-center cell types in motif: {non_center_types}")
        print("="*80)

        # Get neighbor and non-neighbor cell IDs using original motif
        neighbor_result = spatial_utils.get_motif_neighbor_cells(sq_obj=self, ct=ct, motif=motif, max_dist=max_dist, k=k, min_size=min_size)

        # Extract paired data
        center_neighbor_pairs = neighbor_result['center_neighbor_pairs']
        ct_in_motif = neighbor_result['ct_in_motif']

        # Extract unique center and neighbor cells from pairs
        center_cells = np.unique(center_neighbor_pairs[:, 0])
        neighbor_cells = np.unique(center_neighbor_pairs[:, 1])

        # Get all motif cells
        motif_mask = np.isin(np.array(self.labels), motif)
        all_motif_cells = np.where(motif_mask)[0]
        non_neighbor_cells = np.setdiff1d(all_motif_cells, neighbor_cells)

        # Remove center type cells from non-neighbor cells
        if ct_in_motif:
            center_cell_mask_non = self.labels[non_neighbor_cells] == ct
            non_neighbor_cells = non_neighbor_cells[~center_cell_mask_non]
            print(f'Removed {center_cell_mask_non.sum()} center type cells from non-neighbor group.')

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

        # Filter genes by non-zero expression
        if is_sparse:
            nonzero_all = np.array((expr_genes > 0).sum(axis=0)).flatten()
        else:
            nonzero_all = (expr_genes > 0).sum(axis=0)

        valid_gene_mask = nonzero_all >= min_nonzero

        if valid_gene_mask.sum() == 0:
            raise ValueError(f"No genes passed the min_nonzero={min_nonzero} filter.")

        # Apply gene filter
        filtered_genes = [valid_genes[i] for i in range(len(valid_genes)) if valid_gene_mask[i]]
        expr_genes = expr_genes[:, valid_gene_mask]

        print(f"After filtering (min_nonzero={min_nonzero}): {len(filtered_genes)} genes")

        # Compute cell type means for ALL cell types
        all_cell_types = np.unique(self.labels)
        cell_type_means = {}

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
        n_genes = len(filtered_genes)

        # ==================================================================================
        # Correlation 3: Center without motif vs Neighbors (SAME FOR ALL TYPES)
        # ==================================================================================
        print("\n" + "="*80)
        print("Computing Correlation-3: Center without motif vs Neighbors")
        print("="*80)

        start_time = time()
        no_motif_result = spatial_utils.get_all_neighbor_cells(sq_obj=self,
            ct=ct,
            max_dist=max_dist,
            k=k,
            min_size=min_size,
            exclude_centers=center_cells,
            exclude_neighbors=neighbor_cells,
        )

        center_no_motif_pairs = no_motif_result['center_neighbor_pairs']

        if len(center_no_motif_pairs) < 10:
            print(f"Not enough pairs ({len(center_no_motif_pairs)}) for centers without motif. Skipping Correlation 3.")
            corr_matrix_no_motif = None
            n_eff_no_motif = 0
        else:

            # Compute correlation 3 (same logic as original function)
            pair_centers_no_motif = center_no_motif_pairs[:, 0]
            pair_neighbors_no_motif = center_no_motif_pairs[:, 1]

            neighbor_no_motif_cell_types = self.labels[pair_neighbors_no_motif]
            
            # Use helper function for correlation computation (paired data)
            corr_matrix_no_motif, n_eff_no_motif = spatial_utils.compute_cross_correlation_paired(
                sq_obj=self,
                expr_genes=expr_genes,
                pair_centers=pair_centers_no_motif,
                pair_neighbors=pair_neighbors_no_motif,
                center_mean=center_mean,
                cell_type_means=cell_type_means,
                neighbor_cell_types=neighbor_no_motif_cell_types,
                is_sparse=is_sparse
            )
            print(f"Unique center cells: {len(np.unique(pair_centers_no_motif))}")
            print(f"Unique neighbor cells: {len(np.unique(pair_neighbors_no_motif))}")
            print(f"Number of pairs: {len(pair_centers_no_motif)}")
            print(f"Effective sample size: {n_eff_no_motif}")

        end_time = time()
        print(f"Time for computing Correlation 3: {end_time - start_time:.2f} seconds")

        # ==================================================================================
        # Compute correlations for each cell type separately
        # ==================================================================================
        all_results = []

        for cell_type in non_center_types:
            print("\n" + "="*80)
            print(f"Processing cell type: {cell_type}")
            print("="*80)

            # Filter pairs and cells for this specific cell type
            pair_neighbors = center_neighbor_pairs[:, 1]
            neighbor_types = self.labels[pair_neighbors]
            type_mask = neighbor_types == cell_type

            if type_mask.sum() == 0:
                raise ValueError(f"Error: No neighbor pairs found for cell type {cell_type}. Shouldn't happen.")

            type_specific_pairs = center_neighbor_pairs[type_mask]
            type_specific_neighbor_cells = np.unique(type_specific_pairs[:, 1])
            type_specific_center_cells = np.unique(type_specific_pairs[:, 0])

            # Filter non-neighbor cells for this type
            type_non_neighbor_mask = self.labels[non_neighbor_cells] == cell_type
            type_non_neighbor_cells = non_neighbor_cells[type_non_neighbor_mask]

            if len(type_non_neighbor_cells) < 10:
                print(f"Not enough non-neighbor cells ({len(type_non_neighbor_cells)}) for {cell_type}. Skipping.")
                continue
            
            print(f"{len(type_specific_center_cells)} center cells paired with {cell_type}")
            print(f"{len(type_specific_neighbor_cells)} neighboring cells of {cell_type}")
            print(f"{len(type_non_neighbor_cells)} distant cells of {cell_type}")
            print(f"{len(type_specific_pairs)} center-neighboring {cell_type} pairs")

            # ==================================================================================
            # Correlation 1: Center with motif vs Neighboring cells of THIS TYPE (PAIRED)
            # ==================================================================================
            print(f"\nComputing Correlation-1...")
            start_time = time()

            pair_centers = type_specific_pairs[:, 0]
            pair_neighbors_idx = type_specific_pairs[:, 1]

            # All neighbors are of the same type, create uniform type array
            neighbor_types_uniform = np.full(len(pair_neighbors_idx), cell_type)

            corr_matrix_neighbor, n_eff_neighbor = spatial_utils.compute_cross_correlation_paired(
                sq_obj=self,
                expr_genes=expr_genes,
                pair_centers=pair_centers,
                pair_neighbors=pair_neighbors_idx,
                center_mean=center_mean,
                cell_type_means=cell_type_means,
                neighbor_cell_types=neighbor_types_uniform,
                is_sparse=is_sparse
            )

            print(f"  Effective sample size: {n_eff_neighbor}")
            print(f"  Time: {time() - start_time:.2f} seconds")

            # ==================================================================================
            # Correlation 2: Center with motif vs Distant cells of THIS TYPE (ALL PAIRS)
            # ==================================================================================
            print(f"\nComputing Correlation-2 ...")
            start_time = time()

            n_non_neighbor = len(type_non_neighbor_cells)

            # All non-neighbors are of the same type, create uniform type array
            non_neighbor_types_uniform = np.full(n_non_neighbor, cell_type)

            corr_matrix_non_neighbor, n_eff_non_neighbor = spatial_utils.compute_cross_correlation_all_to_all(
                sq_obj=self,
                expr_genes=expr_genes,
                center_cells=type_specific_center_cells,
                non_neighbor_cells=type_non_neighbor_cells,
                center_mean=center_mean,
                cell_type_means=cell_type_means,
                non_neighbor_cell_types=non_neighbor_types_uniform,
                is_sparse=is_sparse
            )

            print(f"  Effective sample size: {n_eff_non_neighbor}")
            print(f"  Time: {time() - start_time:.2f} seconds")

            # ==================================================================================
            # Statistical testing
            # ==================================================================================
            print(f"\nPerforming statistical tests for {cell_type}...")
            start_time = time()

            # Test 1: Corr1 vs Corr2
            _, p_value_test1 = spatial_utils.fisher_z_test(
                corr_matrix_neighbor, n_eff_neighbor,
                corr_matrix_non_neighbor, n_eff_non_neighbor
            )
            delta_corr_test1 = corr_matrix_neighbor - corr_matrix_non_neighbor

            # Test 2: Corr1 vs Corr3
            if corr_matrix_no_motif is not None:
                _, p_value_test2 = spatial_utils.fisher_z_test(
                    corr_matrix_neighbor, n_eff_neighbor,
                    corr_matrix_no_motif, n_eff_no_motif
                )
                delta_corr_test2 = corr_matrix_neighbor - corr_matrix_no_motif
                combined_score = (0.3 * delta_corr_test1 * (-np.log10(p_value_test1 + 1e-300)) +
                                0.7 * delta_corr_test2 * (-np.log10(p_value_test2 + 1e-300)))
            else:
                p_value_test2 = None
                delta_corr_test2 = None
                combined_score = None

            # Create results for this cell type
            gene_center_idx, gene_motif_idx = np.meshgrid(np.arange(n_genes), np.arange(n_genes), indexing='ij')

            type_results_df = pd.DataFrame({
                'cell_type': cell_type,
                'gene_center': np.array(filtered_genes)[gene_center_idx.flatten()],
                'gene_motif': np.array(filtered_genes)[gene_motif_idx.flatten()],
                'corr_neighbor': corr_matrix_neighbor.flatten(),
                'corr_non_neighbor': corr_matrix_non_neighbor.flatten(),
                'p_value_test1': p_value_test1.flatten(),
                'delta_corr_test1': delta_corr_test1.flatten(),
            })

            if corr_matrix_no_motif is not None:
                type_results_df['corr_center_no_motif'] = corr_matrix_no_motif.flatten()
                type_results_df['p_value_test2'] = p_value_test2.flatten()
                type_results_df['delta_corr_test2'] = delta_corr_test2.flatten()
                type_results_df['combined_score'] = combined_score.flatten()
            else:
                type_results_df['corr_center_no_motif'] = np.nan
                type_results_df['p_value_test2'] = np.nan
                type_results_df['delta_corr_test2'] = np.nan
                type_results_df['combined_score'] = np.nan

            all_results.append(type_results_df)

            print(f"  Time: {time() - start_time:.2f} seconds")

        # ==================================================================================
        # Combine results from all cell types and apply FDR correction
        # ==================================================================================
        print("\n" + "="*80)
        print("Combining results and applying FDR correction")
        print("="*80)

        if len(all_results) == 0:
            raise ValueError("No results generated for any cell type. Check input parameters.")

        combined_results = pd.concat(all_results, ignore_index=True)

        print(f"Total gene pairs across all cell types: {len(combined_results)}")

        if corr_matrix_no_motif is not None:
            # Filter by direction consistency
            same_direction = np.sign(combined_results['delta_corr_test1']) == np.sign(combined_results['delta_corr_test2'])
            print(f"Gene pairs with consistent covarying direction: {same_direction.sum()}")

            if same_direction.sum() > 0:
                # Pool all p-values for FDR correction
                combined_results = combined_results[same_direction]
                p_values_test1 = combined_results['p_value_test1'].values
                p_values_test2 = combined_results['p_value_test2'].values

                all_p_values = np.concatenate([p_values_test1, p_values_test2])
                n_consistent = same_direction.sum()

                print(f"Applying FDR correction to {len(all_p_values)} total tests (2 × {n_consistent} gene pairs)")
                alpha = 0.05
                reject_all, q_values_all, _, _ = multipletests(
                    all_p_values,
                    alpha=alpha,
                    method='fdr_bh'
                )

                q_values_test1 = q_values_all[:n_consistent]
                q_values_test2 = q_values_all[n_consistent:]
                reject_test1 = reject_all[:n_consistent]
                reject_test2 = reject_all[n_consistent:]

                combined_results['q_value_test1'] = q_values_test1
                combined_results['q_value_test2'] = q_values_test2
                combined_results['reject_test1_fdr'] = reject_test1
                combined_results['reject_test2_fdr'] = reject_test2

                mask_both_fdr = reject_test1 & reject_test2
                n_both_fdr = mask_both_fdr.sum()

                print(f"Test1 FDR significant (q < {alpha}): {reject_test1.sum()}")
                print(f"Test2 FDR significant (q < {alpha}): {reject_test2.sum()}")
                print(f"Both tests FDR significant: {n_both_fdr}")

                # Summary by cell type
                print(f"\nSignificant gene pairs by cell type:")
                for cell_type in non_center_types:
                    type_mask = combined_results['cell_type'] == cell_type
                    if type_mask.sum() > 0:
                        type_sig = (combined_results.loc[type_mask, 'reject_test1_fdr'] &
                                   combined_results.loc[type_mask, 'reject_test2_fdr']).sum()
                        print(f"  - {cell_type}: {type_sig} significant pairs")
            else:
                combined_results['q_value_test1'] = np.nan
                combined_results['q_value_test2'] = np.nan
                combined_results['reject_test1_fdr'] = False
                combined_results['reject_test2_fdr'] = False
                print("No gene pairs with consistent direction found.")
        else:
            # Only test1 available
            alpha = 0.05
            print("Note: Test2 not available (no centers without motif)")
            p_values_test1_all = combined_results['p_value_test1'].values
            reject_test1, q_values_test1, _, _ = multipletests(
                p_values_test1_all,
                alpha=alpha,
                method='fdr_bh'
            )
            combined_results['q_value_test1'] = q_values_test1
            combined_results['reject_test1_fdr'] = reject_test1
            combined_results['q_value_test2'] = np.nan
            combined_results['reject_test2_fdr'] = False

            print(f"FDR correction applied to {len(combined_results)} gene pairs:")
            print(f"Test1 FDR significant (q < {alpha}): {reject_test1.sum()}")

        # Sort by absolute value of combined score
        if corr_matrix_no_motif is not None:
            combined_results['abs_combined_score'] = np.abs(combined_results['combined_score'])
            combined_results = combined_results.sort_values('abs_combined_score', ascending=False, na_position='last').reset_index(drop=True)
            combined_results['if_significant'] = combined_results['reject_test1_fdr'] & combined_results['reject_test2_fdr']
        else:
            combined_results['test1_score'] = combined_results['delta_corr_test1'] * (-np.log10(combined_results['p_value_test1'] + 1e-300))
            combined_results['combined_score'] = combined_results['test1_score']
            combined_results['abs_combined_score'] = np.abs(combined_results['combined_score'])
            combined_results = combined_results.sort_values('abs_combined_score', ascending=False, na_position='last').reset_index(drop=True)
            combined_results['if_significant'] = combined_results['reject_test1_fdr']
            
        print(f"\nResults prepared and sorted")

        return combined_results

    def compute_gene_gene_correlation_binary(self,
                                              ct: str,
                                              motif: Union[str, List[str]],
                                              genes: Optional[Union[str, List[str]]] = None,
                                              max_dist: Optional[float] = None,
                                              k: Optional[int] = None,
                                              min_size: int = 0,
                                              min_nonzero: int = 10,
                                              ) -> pd.DataFrame:
        """
        Compute gene-gene correlation using deviation from global binary mean for binary expression data.

        Instead of computing phi coefficient directly on binary variables, this method:
        1. Computes global binary expression mean for each gene in each cell type
        2. For each cell, computes deviation = binary_value - global_mean
        3. Computes Pearson correlation between deviation vectors across gene pairs
        4. Compares correlations using Fisher's Z test

        The analysis structure:
        1. Correlation 1: Center with motif vs Neighboring motif (paired)
        2. Correlation 2: Center with motif vs Distant motif (all-to-all)
        3. Correlation 3: Center without motif vs Neighbors (paired)

        Parameters
        ----------
        ct : str
            Cell type as the center cells.
        motif : Union[str, List[str]]
            Motif (names of cell types) to be analyzed.
        genes : Optional[Union[str, List[str]]], default=None
            List of genes to analyze. If None, all genes in the index will be used.
        max_dist : Optional[float], default=None
            Maximum distance for considering a cell as a neighbor. Use either max_dist or k.
        k : Optional[int], default=None
            Number of nearest neighbors. Use either max_dist or k.
        min_size : int, default=0
            Minimum neighborhood size for each center cell.
        min_nonzero : int, default=10
            Minimum number of non-zero expression values required for a gene to be included.

        Returns
        -------
        results_df : pd.DataFrame
            DataFrame with deviation-based correlation results between neighbor and non-neighbor groups.
        cell_groups : dict
            Dictionary containing cell pairing information.
        """
        motif = motif if isinstance(motif, list) else [motif]

        # Get neighbor and non-neighbor cell IDs using the same logic as compute_gene_gene_correlation
        neighbor_result = spatial_utils.get_motif_neighbor_cells(
            sq_obj=self, ct=ct, motif=motif, max_dist=max_dist, k=k, min_size=min_size
        )

        center_neighbor_pairs = neighbor_result['center_neighbor_pairs']
        ct_in_motif = neighbor_result['ct_in_motif']

        # Extract unique center and neighbor cells from pairs
        center_cells = np.unique(center_neighbor_pairs[:, 0])
        neighbor_cells = np.unique(center_neighbor_pairs[:, 1])

        # Get non-neighbor motif cells
        motif_mask = np.isin(self.labels, motif)
        all_motif_cells = np.where(motif_mask)[0]
        non_neighbor_cells = np.setdiff1d(all_motif_cells, neighbor_cells)

        # Remove center type cells from non-neighbor groups
        if ct_in_motif:
            center_cell_mask_non = self.labels[non_neighbor_cells] == ct
            non_neighbor_cells = non_neighbor_cells[~center_cell_mask_non]
            print(f'Focus on inter-cell-type interactions: Remove center type cells from non-neighbor groups.')

        if len(non_neighbor_cells) < 10:
            raise ValueError(f"Not enough non-neighbor cells ({len(non_neighbor_cells)}) for correlation analysis.")

        # Get gene list
        if genes is None:
            valid_genes = self.genes
        elif isinstance(genes, str):
            genes = [genes]
            valid_genes = [g for g in genes if g in self.index.scfindGenes]
        else:
            valid_genes = [g for g in genes if g in self.index.scfindGenes]

        if len(valid_genes) == 0:
            raise ValueError("No valid genes found in the scfind index.")

        print(f"Building binary expression matrix from scfind index for {len(valid_genes)} genes...")
        start_time = time()

        # Use efficient C++ method to get sparse matrix data directly
        # Returns: {'rows': np.array, 'cols': np.array, 'gene_names': list, 'n_cells': int}
        sparse_data = self.index.index.getBinarySparseMatrixData(valid_genes, self.dataset, min_nonzero)

        rows = sparse_data['rows']
        cols = sparse_data['cols']
        filtered_genes = sparse_data['gene_names']
        n_cells = sparse_data['n_cells']

        if len(filtered_genes) == 0:
            raise ValueError(f"No genes passed the min_nonzero={min_nonzero} filter.")

        # Create binary sparse matrix (cells × genes) directly from C++ output
        binary_expr = csr_matrix(
            (np.ones(len(rows), dtype=np.int16), (rows, cols)),
            shape=(n_cells, len(filtered_genes)),
        )
        print(f"Time for recovering binary matrix: {time() - start_time:.2f} seconds")

        # ==================================================================================
        # Compute global binary means for ALL cell types (similar to compute_gene_gene_correlation)
        # ==================================================================================
        print("\n" + "="*60)
        print("Computing global binary expression means for all cell types")
        print("="*60)
        start_time = time()

        # Get all unique cell types in the dataset
        all_cell_types = np.unique(self.labels)
        cell_type_means = {}  # Dictionary to store mean expression for each cell type

        for cell_type in all_cell_types:
            ct_mask = self.labels == cell_type
            ct_cells = np.where(ct_mask)[0]
            if len(ct_cells) > 0:
                ct_expr = binary_expr[ct_cells, :]
                cell_type_means[cell_type] = np.array(ct_expr.mean(axis=0)).flatten()

        center_mean = cell_type_means[ct]
        print(f"\nTime for computing global means: {time() - start_time:.2f} seconds")

        # ==================================================================================
        # Correlation 1: Center with motif vs Neighboring motif (PAIRED)
        # ==================================================================================
        print("\n" + "="*60)
        print("Computing Correlation 1: Center with motif vs Neighboring motif (paired)")
        print("="*60)
        start_time = time()

        pair_centers = center_neighbor_pairs[:, 0]
        pair_neighbors = center_neighbor_pairs[:, 1]

        # Get cell types for each neighbor in pairs
        neighbor_cell_types = self.labels[pair_neighbors]

        corr_matrix_neighbor, n_eff_neighbor = spatial_utils.compute_cross_correlation_paired(
            sq_obj=self,
            expr_genes=binary_expr,
            pair_centers=pair_centers,
            pair_neighbors=pair_neighbors,
            center_mean=center_mean,
            cell_type_means=cell_type_means,
            neighbor_cell_types=neighbor_cell_types,
            is_sparse=True
        )

        print(f"Number of pairs: {len(pair_centers)}")
        print(f"Unique center cells: {len(np.unique(pair_centers))}")
        print(f"Unique neighbor cells: {len(np.unique(pair_neighbors))}")
        print(f"Effective sample size: {n_eff_neighbor}")
        print(f"Time for computing Corr-1: {time() - start_time:.4f} seconds")

        # ==================================================================================
        # Correlation 2: Center with motif vs Distant motif (ALL PAIRS)
        # ==================================================================================
        print("\n" + "="*60)
        print("Computing Correlation 2: Center with motif vs Distant motif (all pairs)")
        print("="*60)
        start_time = time()

        # Get cell types for non-neighbor cells
        non_neighbor_cell_types = self.labels[non_neighbor_cells]

        corr_matrix_non_neighbor, n_eff_non_neighbor = spatial_utils.compute_cross_correlation_all_to_all(
            sq_obj=self,
            expr_genes=binary_expr,
            center_cells=center_cells,
            non_neighbor_cells=non_neighbor_cells,
            center_mean=center_mean,
            cell_type_means=cell_type_means,
            non_neighbor_cell_types=non_neighbor_cell_types,
            is_sparse=True
        )

        print(f"Unique center cells: {len(center_cells)}")
        print(f"Unique distant cells: {len(non_neighbor_cells)}")
        print(f"Effective sample size: {n_eff_non_neighbor}")
        print(f"Time for computing Corr-2: {time() - start_time:.2f} seconds")

        # ==================================================================================
        # Correlation 3: Center without motif vs Neighbors (PAIRED)
        # ==================================================================================
        print("\n" + "="*60)
        print("Computing Correlation 3: Center without motif vs Neighbors (paired)")
        print("="*60)
        start_time = time()

        no_motif_result = spatial_utils.get_all_neighbor_cells(
            sq_obj=self,
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
        else:
            pair_centers_no_motif = center_no_motif_pairs[:, 0]
            pair_neighbors_no_motif = center_no_motif_pairs[:, 1]

            # Get cell types for each neighbor in no-motif pairs
            neighbor_cell_types_no_motif = self.labels[pair_neighbors_no_motif]

            corr_matrix_no_motif, n_eff_no_motif = spatial_utils.compute_cross_correlation_paired(
                sq_obj=self,
                expr_genes=binary_expr,
                pair_centers=pair_centers_no_motif,
                pair_neighbors=pair_neighbors_no_motif,
                center_mean=center_mean,
                cell_type_means=cell_type_means,
                neighbor_cell_types=neighbor_cell_types_no_motif,
                is_sparse=True
            )

            print(f"Number of pairs: {len(center_no_motif_pairs)}")
            print(f"Unique center cells: {len(np.unique(pair_centers_no_motif))}")
            print(f"Unique neighbor cells: {len(np.unique(pair_neighbors_no_motif))}")
            print(f"Effective sample size: {n_eff_no_motif}")

        print(f"Time for computing Corr-3: {time() - start_time:.2f} seconds")

        # Prepare results using the same structure as compute_gene_gene_correlation
        print("\n" + "="*60)
        print("Performing Fisher's Z-tests for comparing correlations")
        print("="*60)
        start_time = time()

        # Test 1: Corr1 vs Corr2 (neighbor vs non-neighbor)
        _, p_value_test1 = spatial_utils.fisher_z_test(
            corr_matrix_neighbor, n_eff_neighbor,
            corr_matrix_non_neighbor, n_eff_non_neighbor
        )
        delta_corr_test1 = corr_matrix_neighbor - corr_matrix_non_neighbor

        # Test 2: Corr1 vs Corr3 (neighbor vs no_motif)
        if corr_matrix_no_motif is not None:
            _, p_value_test2 = spatial_utils.fisher_z_test(
                corr_matrix_neighbor, n_eff_neighbor,
                corr_matrix_no_motif, n_eff_no_motif
            )
            delta_corr_test2 = corr_matrix_neighbor - corr_matrix_no_motif

            # Combined score
            combined_score = (0.3 * delta_corr_test1 * (-np.log10(p_value_test1 + 1e-300)) +
                            0.7 * delta_corr_test2 * (-np.log10(p_value_test2 + 1e-300)))
        else:
            p_value_test2 = None
            delta_corr_test2 = None
            combined_score = None

        # Build results DataFrame
        n_genes = len(filtered_genes)
        gene_center_idx, gene_motif_idx = np.meshgrid(np.arange(n_genes), np.arange(n_genes), indexing='ij')

        results_df = pd.DataFrame({
            'gene_center': np.array(filtered_genes)[gene_center_idx.flatten()],
            'gene_motif': np.array(filtered_genes)[gene_motif_idx.flatten()],
            'corr_neighbor': corr_matrix_neighbor.flatten(),
            'corr_non_neighbor': corr_matrix_non_neighbor.flatten(),
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
            results_df['corr_center_no_motif'] = np.nan
            results_df['p_value_test2'] = np.nan
            results_df['delta_corr_test2'] = np.nan
            results_df['combined_score'] = np.nan

        # Apply FDR correction
        print(f"Total gene pairs: {len(results_df)}")

        if corr_matrix_no_motif is not None:
            # Filter by direction consistency
            same_direction = np.sign(results_df['delta_corr_test1']) == np.sign(results_df['delta_corr_test2'])
            print(f"Gene pairs with consistent covarying direction: {same_direction.sum()}")

            if same_direction.sum() > 0:
                p_values_test1 = results_df.loc[same_direction, 'p_value_test1'].values
                p_values_test2 = results_df.loc[same_direction, 'p_value_test2'].values
                all_p_values = np.concatenate([p_values_test1, p_values_test2])
                n_consistent = same_direction.sum()

                print(f"Applying FDR correction to {len(all_p_values)} total tests (2 × {n_consistent} gene pairs)")

                reject_all, q_values_all, _, _ = multipletests(all_p_values, alpha=0.05, method='fdr_bh')

                q_values_test1 = q_values_all[:n_consistent]
                q_values_test2 = q_values_all[n_consistent:]
                reject_test1 = reject_all[:n_consistent]
                reject_test2 = reject_all[n_consistent:]

                results_df.loc[same_direction, 'q_value_test1'] = q_values_test1
                results_df.loc[same_direction, 'q_value_test2'] = q_values_test2
                results_df.loc[same_direction, 'reject_test1_fdr'] = reject_test1
                results_df.loc[same_direction, 'reject_test2_fdr'] = reject_test2

                results_df = results_df[same_direction]

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
            print("Note: Test2 not available (no centers without motif)")
            p_values_test1_all = results_df['p_value_test1'].values
            reject_test1, q_values_test1, _, _ = multipletests(p_values_test1_all, alpha=0.05, method='fdr_bh')
            results_df['q_value_test1'] = q_values_test1
            results_df['reject_test1_fdr'] = reject_test1
            results_df['q_value_test2'] = np.nan
            results_df['reject_test2_fdr'] = False

            print(f"FDR correction applied to all {len(results_df)} gene pairs:")
            print(f"  - Test1 FDR significant (q < 0.05): {reject_test1.sum()}")

        # Sort by absolute value of combined score
        if corr_matrix_no_motif is not None:
            results_df['abs_combined_score'] = np.abs(results_df['combined_score'])
            results_df = results_df.sort_values('abs_combined_score', ascending=False, na_position='last').reset_index(drop=True)
            results_df['if_significant'] = results_df['reject_test1_fdr'] & results_df['reject_test2_fdr']
        else:
            results_df['test1_score'] = results_df['delta_corr_test1'] * (-np.log10(results_df['p_value_test1'] + 1e-300))
            results_df['combined_score'] = results_df['test1_score']
            results_df['abs_combined_score'] = np.abs(results_df['combined_score'])
            results_df = results_df.sort_values('abs_combined_score', ascending=False, na_position='last')
            results_df['if_significant'] = results_df['reject_test1_fdr'] & results_df['reject_test2_fdr']

        print(f"Time for testing results: {time() - start_time:.2f} seconds\n")

        # Prepare cell groups dictionary
        cell_groups = {
            'center_neighbor_motif_pair': center_neighbor_pairs,
            'non-neighbor_motif_cells': non_neighbor_cells,
            'non_motif_center_neighbor_pair': center_no_motif_pairs if corr_matrix_no_motif is not None else np.array([]).reshape(0, 2),
        }

        return results_df, cell_groups

    def compute_gene_gene_correlation_binary_by_type(self,
                                                     ct: str,
                                                     motif: Union[str, List[str]],
                                                     genes: Optional[Union[str, List[str]]] = None,
                                                     max_dist: Optional[float] = None,
                                                     k: Optional[int] = None,
                                                     min_size: int = 0,
                                                     min_nonzero: int = 10,
                                                     ) -> pd.DataFrame:
        """
        Compute gene-gene correlation using deviation from global binary mean, separately for each cell type in the motif.

        For each non-center cell type in the motif, compute:
        - Correlation 1: Center cells with motif vs neighboring motif cells of THIS TYPE
        - Correlation 2: Center cells with motif vs distant motif cells of THIS TYPE
        - Correlation 3: Center cells without motif vs neighbors (same for all types)

        Only analyzes motifs with >= 2 cell types besides the center type.

        Parameters
        ----------
        ct : str
            Cell type as the center cells.
        motif : Union[str, List[str]]
            Motif (names of cell types) to be analyzed.
        genes : Optional[Union[str, List[str]]], default=None
            List of genes to analyze. If None, all genes in the index will be used.
        max_dist : Optional[float], default=None
            Maximum distance for considering a cell as a neighbor. Use either max_dist or k.
        k : Optional[int], default=None
            Number of nearest neighbors. Use either max_dist or k.
        min_size : int, default=0
            Minimum neighborhood size for each center cell.
        min_nonzero : int, default=10
            Minimum number of non-zero expression values required for a gene to be included.

        Returns
        -------
        results_df : pd.DataFrame
            DataFrame with correlation results for each cell type and gene pair.
            Columns include:
                - cell_type: the non-center cell type in motif
                - gene_center, gene_motif: gene pairs
                - corr_neighbor: correlation with neighboring cells of this type
                - corr_non_neighbor: correlation with distant cells of this type
                - corr_center_no_motif: correlation with neighbors when no motif present
                - p_value_test1, p_value_test2: p-values for statistical tests
                - q_value_test1, q_value_test2: FDR-corrected q-values
                - delta_corr_test1, delta_corr_test2: correlation differences
                - reject_test1_fdr, reject_test2_fdr: FDR significance flags
                - combined_score, abs_combined_score: combined significance scores
                - if_significant: whether both tests pass FDR threshold
        """
        motif = motif if isinstance(motif, list) else [motif]

        # Get non-center cell types in motif
        non_center_types = [m for m in motif if m != ct]

        if len(non_center_types) == 1:
            print(f"Only one non-center cell type in motif: {non_center_types}. Using compute_gene_gene_correlation_binary method.")
            result, _ = self.compute_gene_gene_correlation_binary(
                ct=ct,
                motif=motif,
                genes=genes,
                max_dist=max_dist,
                k=k,
                min_size=min_size,
                min_nonzero=min_nonzero
            )
            return result
        elif len(non_center_types) == 0:
            raise ValueError("Error: Only center cell type in motif. Please ensure motif includes at least one non-center cell type.")

        print(f"Analyzing {len(non_center_types)} non-center cell types in motif: {non_center_types}")
        print("="*80)

        # Get neighbor and non-neighbor cell IDs using original motif
        neighbor_result = spatial_utils.get_motif_neighbor_cells(
            sq_obj=self, ct=ct, motif=motif, max_dist=max_dist, k=k, min_size=min_size
        )

        center_neighbor_pairs = neighbor_result['center_neighbor_pairs']
        ct_in_motif = neighbor_result['ct_in_motif']

        # Extract unique center and neighbor cells from pairs
        center_cells = np.unique(center_neighbor_pairs[:, 0])
        neighbor_cells = np.unique(center_neighbor_pairs[:, 1])

        # Get all motif cells
        motif_mask = np.isin(self.labels, motif)
        all_motif_cells = np.where(motif_mask)[0]
        non_neighbor_cells = np.setdiff1d(all_motif_cells, neighbor_cells)

        # Remove center type cells from non-neighbor cells
        if ct_in_motif:
            center_cell_mask_non = self.labels[non_neighbor_cells] == ct
            non_neighbor_cells = non_neighbor_cells[~center_cell_mask_non]
            print(f'Removed {center_cell_mask_non.sum()} center type cells from non-neighbor group.')

        # Get gene list
        if genes is None:
            valid_genes = self.genes
        elif isinstance(genes, str):
            genes = [genes]
            valid_genes = [g for g in genes if g in self.index.scfindGenes]
        else:
            valid_genes = [g for g in genes if g in self.index.scfindGenes]

        if len(valid_genes) == 0:
            raise ValueError("No valid genes found in the scfind index.")

        print(f"Building binary expression matrix from scfind index for {len(valid_genes)} genes...")
        start_time = time()

        # Use efficient C++ method to get sparse matrix data directly
        sparse_data = self.index.index.getBinarySparseMatrixData(valid_genes, self.dataset, min_nonzero)

        rows = sparse_data['rows']
        cols = sparse_data['cols']
        filtered_genes = sparse_data['gene_names']
        n_cells = sparse_data['n_cells']

        if len(filtered_genes) == 0:
            raise ValueError(f"No genes passed the min_nonzero={min_nonzero} filter.")

        # Create binary sparse matrix (cells × genes)
        binary_expr = csr_matrix(
            (np.ones(len(rows),), (rows, cols)),
            shape=(n_cells, len(filtered_genes)),
        )
        print(f"Time for recovering binary matrix: {time() - start_time:.2f} seconds")

        # Compute global binary means for ALL cell types
        print("\n" + "="*80)
        print("Computing global binary expression means for all cell types")
        print("="*80)
        start_time = time()

        all_cell_types = np.unique(self.labels)
        cell_type_means = {}

        for cell_type in all_cell_types:
            ct_mask = self.labels == cell_type
            ct_cells = np.where(ct_mask)[0]
            if len(ct_cells) > 0:
                ct_expr = binary_expr[ct_cells, :]
                cell_type_means[cell_type] = np.array(ct_expr.mean(axis=0)).flatten()

        center_mean = cell_type_means[ct]
        n_genes = len(filtered_genes)
        print(f"\nTime for computing global means: {time() - start_time:.2f} seconds")

        # ==================================================================================
        # Correlation 3: Center without motif vs Neighbors (SAME FOR ALL TYPES)
        # ==================================================================================
        print("\n" + "="*80)
        print("Computing Correlation-3: Center without motif vs Neighbors")
        print("="*80)

        start_time = time()
        no_motif_result = spatial_utils.get_all_neighbor_cells(
            sq_obj=self,
            ct=ct,
            max_dist=max_dist,
            k=k,
            min_size=min_size,
            exclude_centers=center_cells,
            exclude_neighbors=neighbor_cells,
        )

        center_no_motif_pairs = no_motif_result['center_neighbor_pairs']

        if len(center_no_motif_pairs) < 10:
            print(f"Not enough pairs ({len(center_no_motif_pairs)}) for centers without motif. Skipping Correlation 3.")
            corr_matrix_no_motif = None
            n_eff_no_motif = 0
        else:
            pair_centers_no_motif = center_no_motif_pairs[:, 0]
            pair_neighbors_no_motif = center_no_motif_pairs[:, 1]

            neighbor_no_motif_cell_types = self.labels[pair_neighbors_no_motif]

            corr_matrix_no_motif, n_eff_no_motif = spatial_utils.compute_cross_correlation_paired(
                sq_obj=self,
                expr_genes=binary_expr,
                pair_centers=pair_centers_no_motif,
                pair_neighbors=pair_neighbors_no_motif,
                center_mean=center_mean,
                cell_type_means=cell_type_means,
                neighbor_cell_types=neighbor_no_motif_cell_types,
                is_sparse=True
            )
            print(f"Unique center cells: {len(np.unique(pair_centers_no_motif))}")
            print(f"Unique neighbor cells: {len(np.unique(pair_neighbors_no_motif))}")
            print(f"Number of pairs: {len(pair_centers_no_motif)}")
            print(f"Effective sample size: {n_eff_no_motif}")

        end_time = time()
        print(f"Time for computing Correlation 3: {end_time - start_time:.2f} seconds")

        # ==================================================================================
        # Compute correlations for each cell type separately
        # ==================================================================================
        all_results = []

        for cell_type in non_center_types:
            print("\n" + "="*80)
            print(f"Processing cell type: {cell_type}")
            print("="*80)

            # Filter pairs and cells for this specific cell type
            pair_neighbors = center_neighbor_pairs[:, 1]
            neighbor_types = self.labels[pair_neighbors]
            type_mask = neighbor_types == cell_type

            if type_mask.sum() == 0:
                raise ValueError(f"Error: No neighbor pairs found for cell type {cell_type}. Shouldn't happen.")

            type_specific_pairs = center_neighbor_pairs[type_mask]
            type_specific_neighbor_cells = np.unique(type_specific_pairs[:, 1])
            type_specific_center_cells = np.unique(type_specific_pairs[:, 0])

            # Filter non-neighbor cells for this type
            type_non_neighbor_mask = self.labels[non_neighbor_cells] == cell_type
            type_non_neighbor_cells = non_neighbor_cells[type_non_neighbor_mask]

            if len(type_non_neighbor_cells) < 10:
                print(f"Not enough non-neighbor cells ({len(type_non_neighbor_cells)}) for {cell_type}. Skipping.")
                continue

            print(f"{len(type_specific_center_cells)} center cells paired with {cell_type}")
            print(f"{len(type_specific_neighbor_cells)} neighboring cells of {cell_type}")
            print(f"{len(type_non_neighbor_cells)} distant cells of {cell_type}")
            print(f"{len(type_specific_pairs)} center-neighboring {cell_type} pairs")

            # ==================================================================================
            # Correlation 1: Center with motif vs Neighboring cells of THIS TYPE (PAIRED)
            # ==================================================================================
            print(f"\nComputing Correlation-1...")
            start_time = time()

            pair_centers = type_specific_pairs[:, 0]
            pair_neighbors_idx = type_specific_pairs[:, 1]

            # All neighbors are of the same type, create uniform type array
            neighbor_types_uniform = np.full(len(pair_neighbors_idx), cell_type)

            corr_matrix_neighbor, n_eff_neighbor = spatial_utils.compute_cross_correlation_paired(
                sq_obj=self,
                expr_genes=binary_expr,
                pair_centers=pair_centers,
                pair_neighbors=pair_neighbors_idx,
                center_mean=center_mean,
                cell_type_means=cell_type_means,
                neighbor_cell_types=neighbor_types_uniform,
                is_sparse=True
            )

            print(f"  Effective sample size: {n_eff_neighbor}")
            print(f"  Time: {time() - start_time:.2f} seconds")

            # ==================================================================================
            # Correlation 2: Center with motif vs Distant cells of THIS TYPE (ALL PAIRS)
            # ==================================================================================
            print(f"\nComputing Correlation-2...")
            start_time = time()

            n_non_neighbor = len(type_non_neighbor_cells)

            # All non-neighbors are of the same type
            non_neighbor_types_uniform = np.full(n_non_neighbor, cell_type)

            corr_matrix_non_neighbor, n_eff_non_neighbor = spatial_utils.compute_cross_correlation_all_to_all(
                sq_obj=self,
                expr_genes=binary_expr,
                center_cells=type_specific_center_cells,
                non_neighbor_cells=type_non_neighbor_cells,
                center_mean=center_mean,
                cell_type_means=cell_type_means,
                non_neighbor_cell_types=non_neighbor_types_uniform,
                is_sparse=True
            )

            print(f"  Effective sample size: {n_eff_non_neighbor}")
            print(f"  Time: {time() - start_time:.2f} seconds")

            # ==================================================================================
            # Statistical testing
            # ==================================================================================
            print(f"\nPerforming statistical tests for {cell_type}...")
            start_time = time()

            # Test 1: Corr1 vs Corr2
            _, p_value_test1 = spatial_utils.fisher_z_test(
                corr_matrix_neighbor, n_eff_neighbor,
                corr_matrix_non_neighbor, n_eff_non_neighbor
            )
            delta_corr_test1 = corr_matrix_neighbor - corr_matrix_non_neighbor

            # Test 2: Corr1 vs Corr3
            if corr_matrix_no_motif is not None:
                _, p_value_test2 = spatial_utils.fisher_z_test(
                    corr_matrix_neighbor, n_eff_neighbor,
                    corr_matrix_no_motif, n_eff_no_motif
                )
                delta_corr_test2 = corr_matrix_neighbor - corr_matrix_no_motif
                combined_score = (0.3 * delta_corr_test1 * (-np.log10(p_value_test1 + 1e-300)) +
                                0.7 * delta_corr_test2 * (-np.log10(p_value_test2 + 1e-300)))
            else:
                p_value_test2 = None
                delta_corr_test2 = None
                combined_score = None

            # Create results for this cell type
            gene_center_idx, gene_motif_idx = np.meshgrid(np.arange(n_genes), np.arange(n_genes), indexing='ij')

            type_results_df = pd.DataFrame({
                'cell_type': cell_type,
                'gene_center': np.array(filtered_genes)[gene_center_idx.flatten()],
                'gene_motif': np.array(filtered_genes)[gene_motif_idx.flatten()],
                'corr_neighbor': corr_matrix_neighbor.flatten(),
                'corr_non_neighbor': corr_matrix_non_neighbor.flatten(),
                'p_value_test1': p_value_test1.flatten(),
                'delta_corr_test1': delta_corr_test1.flatten(),
            })

            if corr_matrix_no_motif is not None:
                type_results_df['corr_center_no_motif'] = corr_matrix_no_motif.flatten()
                type_results_df['p_value_test2'] = p_value_test2.flatten()
                type_results_df['delta_corr_test2'] = delta_corr_test2.flatten()
                type_results_df['combined_score'] = combined_score.flatten()
            else:
                type_results_df['corr_center_no_motif'] = np.nan
                type_results_df['p_value_test2'] = np.nan
                type_results_df['delta_corr_test2'] = np.nan
                type_results_df['combined_score'] = np.nan

            all_results.append(type_results_df)

            print(f"  Time: {time() - start_time:.2f} seconds")

        # ==================================================================================
        # Combine results from all cell types and apply FDR correction
        # ==================================================================================
        print("\n" + "="*80)
        print("Combining results and applying FDR correction")
        print("="*80)

        if len(all_results) == 0:
            raise ValueError("No results generated for any cell type. Check input parameters.")

        combined_results = pd.concat(all_results, ignore_index=True)

        print(f"Total gene pairs across all cell types: {len(combined_results)}")

        if corr_matrix_no_motif is not None:
            # Filter by direction consistency
            same_direction = np.sign(combined_results['delta_corr_test1']) == np.sign(combined_results['delta_corr_test2'])
            print(f"Gene pairs with consistent covarying direction: {same_direction.sum()}")

            if same_direction.sum() > 0:
                # Pool all p-values for FDR correction
                combined_results = combined_results[same_direction]
                p_values_test1 = combined_results['p_value_test1'].values
                p_values_test2 = combined_results['p_value_test2'].values

                all_p_values = np.concatenate([p_values_test1, p_values_test2])
                n_consistent = same_direction.sum()

                print(f"Applying FDR correction to {len(all_p_values)} total tests (2 × {n_consistent} gene pairs)")
                alpha = 0.05
                reject_all, q_values_all, _, _ = multipletests(
                    all_p_values,
                    alpha=alpha,
                    method='fdr_bh'
                )

                q_values_test1 = q_values_all[:n_consistent]
                q_values_test2 = q_values_all[n_consistent:]
                reject_test1 = reject_all[:n_consistent]
                reject_test2 = reject_all[n_consistent:]

                combined_results['q_value_test1'] = q_values_test1
                combined_results['q_value_test2'] = q_values_test2
                combined_results['reject_test1_fdr'] = reject_test1
                combined_results['reject_test2_fdr'] = reject_test2

                mask_both_fdr = reject_test1 & reject_test2
                n_both_fdr = mask_both_fdr.sum()

                print(f"Test1 FDR significant (q < {alpha}): {reject_test1.sum()}")
                print(f"Test2 FDR significant (q < {alpha}): {reject_test2.sum()}")
                print(f"Both tests FDR significant: {n_both_fdr}")

                # Summary by cell type
                print(f"\nSignificant gene pairs by cell type:")
                for cell_type in non_center_types:
                    type_mask = combined_results['cell_type'] == cell_type
                    if type_mask.sum() > 0:
                        type_sig = (combined_results.loc[type_mask, 'reject_test1_fdr'] &
                                   combined_results.loc[type_mask, 'reject_test2_fdr']).sum()
                        print(f"  - {cell_type}: {type_sig} significant pairs")
            else:
                combined_results['q_value_test1'] = np.nan
                combined_results['q_value_test2'] = np.nan
                combined_results['reject_test1_fdr'] = False
                combined_results['reject_test2_fdr'] = False
                print("No gene pairs with consistent direction found.")
        else:
            # Only test1 available
            alpha = 0.05
            print("Note: Test2 not available (no centers without motif)")
            p_values_test1_all = combined_results['p_value_test1'].values
            reject_test1, q_values_test1, _, _ = multipletests(
                p_values_test1_all,
                alpha=alpha,
                method='fdr_bh'
            )
            combined_results['q_value_test1'] = q_values_test1
            combined_results['reject_test1_fdr'] = reject_test1
            combined_results['q_value_test2'] = np.nan
            combined_results['reject_test2_fdr'] = False

            print(f"FDR correction applied to {len(combined_results)} gene pairs:")
            print(f"Test1 FDR significant (q < {alpha}): {reject_test1.sum()}")

        # Sort by absolute value of combined score
        if corr_matrix_no_motif is not None:
            combined_results['abs_combined_score'] = np.abs(combined_results['combined_score'])
            combined_results = combined_results.sort_values('abs_combined_score', ascending=False, na_position='last').reset_index(drop=True)
            combined_results['if_significant'] = combined_results['reject_test1_fdr'] & combined_results['reject_test2_fdr']
        else:
            combined_results['test1_score'] = combined_results['delta_corr_test1'] * (-np.log10(combined_results['p_value_test1'] + 1e-300))
            combined_results['combined_score'] = combined_results['test1_score']
            combined_results['abs_combined_score'] = np.abs(combined_results['combined_score'])
            combined_results = combined_results.sort_values('abs_combined_score', ascending=False, na_position='last').reset_index(drop=True)
            combined_results['if_significant'] = combined_results['reject_test1_fdr']

        print(f"\nResults prepared and sorted")

        return combined_results

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
        Test whether gene-pairs have significantly different correlation scores between two groups.

        Parameters
        ----------
        result_A : pd.DataFrame
            Results from compute_gene_gene_correlation/_by_type for condition A
            Must contain columns: gene_center, gene_motif, combined_score, if_significant
        result_B : pd.DataFrame
            Results from compute_gene_gene_correlation/_by_type for condition B
            Must contain the same columns as result_A
        score_col : str, default='combined_score'
            Name of the column containing correlation scores to compare
        significance_col : str, default='if_significant'
            Name of the column indicating whether a pair is significant
        gene_center_col : str, default='gene_center'
            Name of the column containing center gene names
        gene_motif_col : str, default='gene_motif'
            Name of the column containing motif gene names
        percentile_threshold : float, default=95.0
            Percentile threshold for identifying outliers (e.g., 95 means top/bottom 5%)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - gene_center: center gene name
            - gene_motif: motif gene name
            - score_A: score in condition A
            - score_B: score in condition B
            - score_diff: score_A - score_B
            - percentile: percentile rank of score_diff in the distribution
            - is_outlier: whether this pair is an outlier (percentile > 95 or < 5)
            - significant_in_A: whether pair is significant in condition A
            - significant_in_B: whether pair is significant in condition B
            - outlier_direction: 'higher_in_A' (>95th), 'lower_in_A' (<5th), or 'not_outlier'

        """
        from .spatial_utils import test_score_difference
        return test_score_difference(result_A, result_B, score_col, significance_col, gene_center_col, gene_motif_col, percentile_threshold, background)
    
    def plot_fov(self,
                 min_cells_label: int = 50,
                 title: str = 'Spatial distribution of cell types',
                 figsize: tuple = (10, 5),
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
        figsize:
            Figure size parameter.

        save_path:
            Path to save the figure.
            If None, the figure will not be saved.

        Return
        ------
        A figure.
        """
        return plotting.plot_fov(sq_obj=self, min_cells_label=min_cells_label, title=title, figsize=figsize, save_path=save_path)

    def plot_motif_grid(self,
                        motif: Union[str, List[str]],
                        figsize: tuple = (10, 5),
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
                            max_dist: float = 100,
                            figsize: tuple = (5, 5),
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
