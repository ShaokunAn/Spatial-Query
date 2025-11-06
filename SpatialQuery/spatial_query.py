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
import matplotlib.cm as cm

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
        max_radius: float = 500,
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
        Similar to motif_enrichment_* with return_cellID=True, but only returns cell IDs.

        For kNN: filters out neighbors beyond max_dist.
        For dist: filters out center cells with fewer than min_size neighbors.

        If ct is in motif, motif cells of center type are also included in center_id.

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
            'center_id': array of center cell IDs (original centers + motif cells of ct if ct in motif)
            'neighbor_id_all': array of motif cell IDs that are neighbors of ALL center cells
            'neighbor_id_with_motif': array of motif cell IDs that are neighbors of center cells with motif
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

        label_encoder = LabelEncoder()
        int_labels = label_encoder.fit_transform(np.array(self.labels))
        int_ct = label_encoder.transform(np.array(ct, dtype=object, ndmin=1))
        int_motifs = label_encoder.transform(np.array(motif))

        num_cells = len(self.spatial_pos)
        num_types = len(label_encoder.classes_)
        motif_mask = np.isin(np.array(self.labels), motif)

        if max_dist is not None:
            # Distance-based neighbors
            max_dist = min(max_dist, self.max_radius)
            idxs_all = self.kd_tree.query_ball_point(
                self.spatial_pos,
                r=max_dist,
                return_sorted=False,
                workers=-1,
            )
            # Filter out self and apply min_size filter
            idxs_all_filter = [np.array(ids)[np.array(ids) != i] for i, ids in enumerate(idxs_all)]

            # Get all neighbors of all center cells (for computing non-neighbor cells)
            all_center_neighbors = set()
            for cind in cinds:
                if len(idxs_all_filter[cind]) >= min_size:  # Apply min_size filter
                    all_center_neighbors.update(idxs_all_filter[cind])

            # Build neighbor matrix to find centers with motif
            flat_neighbors_all = np.concatenate(idxs_all_filter)
            row_indices_all = np.repeat(np.arange(num_cells), [len(neigh) for neigh in idxs_all_filter])
            neighbor_labels_all = int_labels[flat_neighbors_all]

            neighbor_matrix_all = np.zeros((num_cells, num_types), dtype=int)
            np.add.at(neighbor_matrix_all, (row_indices_all, neighbor_labels_all), 1)

            mask_all = int_labels == int_ct
            inds_all = np.where(np.all(neighbor_matrix_all[mask_all][:, int_motifs] > 0, axis=1))[0]
            cind_with_motif = np.array([cinds[i] for i in inds_all if len(idxs_all_filter[cinds[i]]) >= min_size])

            # Get neighbors of centers with motif
            all_neighbors_with_motif = np.concatenate([idxs_all_filter[i] for i in cind_with_motif])
            valid_neighbors_with_motif = all_neighbors_with_motif[motif_mask[all_neighbors_with_motif]]
            id_motif_celltype_with_motif = np.unique(valid_neighbors_with_motif)

            # Get all neighbors of all center cells (that are motif types)
            id_all_center_neighbors = np.array(list(all_center_neighbors))
            id_motif_all_neighbors = id_all_center_neighbors[motif_mask[id_all_center_neighbors]]

        else:
            # KNN-based neighbors (following motif_enrichment_knn logic with max_dist cutoff)
            dists, idxs = self.kd_tree.query(self.spatial_pos, k=k + 1, workers=-1)

            # Apply distance cutoff (like in motif_enrichment_knn)
            if max_dist is None:
                max_dist = self.max_radius
            max_dist = min(max_dist, self.max_radius)

            valid_neighbors = dists[:, 1:] <= max_dist
            filtered_idxs = np.where(valid_neighbors, idxs[:, 1:], -1)

            # Get all neighbors of all center cells
            all_center_neighbors = set()
            for cind in cinds:
                valid_neighs = filtered_idxs[cind][valid_neighbors[cind, :]]
                all_center_neighbors.update(valid_neighs)

            # Build neighbor matrix
            flat_neighbors = filtered_idxs.flatten()
            valid_neighbors_flat = valid_neighbors.flatten()
            neighbor_labels = np.where(valid_neighbors_flat, int_labels[flat_neighbors], -1)
            valid_mask = neighbor_labels != -1

            neighbor_matrix = np.zeros((num_cells * k, num_types), dtype=int)
            neighbor_matrix[np.arange(len(neighbor_labels))[valid_mask], neighbor_labels[valid_mask]] = 1
            neighbor_counts = neighbor_matrix.reshape(num_cells, k, num_types).sum(axis=1)

            mask = int_labels == int_ct
            inds = np.where(np.all(neighbor_counts[mask][:, int_motifs] > 0, axis=1))[0]
            cind_with_motif = np.array(cinds)[inds]

            # Get neighbors of centers with motif
            valid_idxs_of_centers = [
                idxs[c, 1:][valid_neighbors[c, :]]
                for c in cind_with_motif
            ]
            valid_neighbors_flat = np.concatenate(valid_idxs_of_centers)
            valid_motif_neighbors = valid_neighbors_flat[motif_mask[valid_neighbors_flat]]
            id_motif_celltype_with_motif = np.unique(valid_motif_neighbors)

            # Get all neighbors of all center cells (that are motif types)
            id_all_center_neighbors = np.array(list(all_center_neighbors))
            id_motif_all_neighbors = id_all_center_neighbors[motif_mask[id_all_center_neighbors]]

        # If ct is in motif, add motif cells of center type to center_id
        final_center_id = cind_with_motif
        if ct_in_motif:
            # Find motif cells that are of center type
            print("Including neighboring cells of center type into center cells.")
            motif_ct_cells = id_motif_celltype_with_motif[self.labels[id_motif_celltype_with_motif] == ct]
            # Add them to center_id
            final_center_id = np.unique(np.concatenate([cind_with_motif, motif_ct_cells]))

        return {
            'center_id': final_center_id,
            'neighbor_id_all': id_motif_all_neighbors,
            'neighbor_id_with_motif': id_motif_celltype_with_motif
        }

    def get_all_neighbor_cells(self,
                               ct: str,
                               max_dist: Optional[float] = None,
                               k: Optional[int] = None,
                               min_size: int = 0,
                               exclude_centers: Optional[np.ndarray] = None,
                               ) -> dict:
        """
        Get all neighbor cells (not limited to motif) for given center cell type.
        Similar to get_motif_neighbor_cells but returns ALL neighbors regardless of cell type.

        Only returns neighbors that are different from center cell type.

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

        Return
        ------
        dict with keys:
            'center_id': array of center cell IDs excluding specified centers
            'neighbor_id': array of all neighbor cell IDs of center_id cells (excluding center type cells)
        """
        if (max_dist is None and k is None) or (max_dist is not None and k is not None):
            raise ValueError("Please specify either max_dist or k, but not both.")

        if ct not in self.labels.unique():
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        cinds = np.where(self.labels == ct)[0]

        # Exclude specified centers if provided
        if exclude_centers is not None:
            cinds = np.setdiff1d(cinds, exclude_centers)

        if max_dist is not None:
            # Distance-based neighbors
            max_dist = min(max_dist, self.max_radius)
            idxs_all = self.kd_tree.query_ball_point(
                self.spatial_pos,
                r=max_dist,
                return_sorted=False,
                workers=-1,
            )

            # Process all center cells
            # Note: query_ball_point returns variable-length lists, making full vectorization difficult
            # But we can still optimize by using list comprehension and filtering
            center_neighbors = [(cind, np.array([n for n in idxs_all[cind] if n != cind]))
                               for cind in cinds]

            # Filter by min_size
            if min_size > 0:
                valid_center_neighbors = [(cind, neighs) for cind, neighs in center_neighbors
                                         if len(neighs) >= min_size]
            else:
                valid_center_neighbors = center_neighbors

            if len(valid_center_neighbors) > 0:
                final_center_id = np.array([cind for cind, _ in valid_center_neighbors])
                all_neighbors = np.unique(np.concatenate([neighs for _, neighs in valid_center_neighbors]))
            else:
                final_center_id = np.array([])
                all_neighbors = np.array([])

        else:
            # KNN-based neighbors
            dists, idxs = self.kd_tree.query(self.spatial_pos, k=k + 1, workers=-1)

            # Apply distance cutoff
            if max_dist is None:
                max_dist = self.max_radius
            max_dist = min(max_dist, self.max_radius)

            valid_neighbors = dists[:, 1:] <= max_dist

            # Get all neighbors of center cells (vectorized)
            center_mask = np.isin(np.arange(len(self.labels)), cinds)
            center_neighbors_flat = idxs[center_mask, 1:].flatten()
            valid_flat = valid_neighbors[center_mask, :].flatten()

            all_neighbors = np.unique(center_neighbors_flat[valid_flat])
            final_center_id = cinds.copy()

        # Separate neighbors by cell type: keep only non-center-type as neighbors
        if len(all_neighbors) > 0:
            neighbor_is_center_type = self.labels[all_neighbors] == ct
            neighbor_id = all_neighbors[~neighbor_is_center_type]
        else:
            neighbor_id = np.array([])

        return {
            'center_id': final_center_id,
            'neighbor_id': neighbor_id
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
        Compute gene-gene cross correlation between neighbor and non-neighbor motif cells. Only considers inter-cell-type interactions. After finding neighbors using
        the full motif, removes all cells of the center cell type from both neighbor and
        non-neighbor groups. For Pearson correlation, uses shifted correlation (subtract cell type mean) to enable
        comparison across different niches/motifs.

        This function calculates cross correlation between gene expression in:
        1. Motif cells that are neighbors of center cell type (excluding center type cells in neighbor group)
        2. Motif cells that are NOT neighbors of center cell type (excluding center type cells)

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
            Dictionary containing cell IDs for different groups:
                - 'center_with_motif': center cells with motif nearby
                - 'center_without_motif': center cells without motif nearby
                - 'neighbor_motif': motif cells that are neighbors of center cells with motif
                - 'non_neighbor_motif': motif cells that are not neighbors of center cells
                - 'center_without_motif_neighbors': neighbors of center cells without motif
                  (excluding any overlap with neighbor_motif - overlapping cells are attributed
                  to the neighbor_motif group)
        """
        if self.adata is None:
            raise ValueError("Expression data (adata) is not available. Please set build_gene_index=False when initializing spatial_query.")
        
        motif = motif if isinstance(motif, list) else [motif]

        # Get neighbor and non-neighbor cell IDs (using original motif)
        neighbor_result = self.get_motif_neighbor_cells(ct=ct, motif=motif, max_dist=max_dist, k=k, min_size=min_size)
        neighbor_cells = neighbor_result['neighbor_id_with_motif']

        all_neighbor_cells = neighbor_result['neighbor_id_all']

        # Get non-neighbor motif cells (set difference)
        motif_mask = np.isin(np.array(self.labels), motif)
        all_motif_cells = np.where(motif_mask)[0]
        non_neighbor_cells = np.setdiff1d(all_motif_cells, all_neighbor_cells)

        # Remove cells of center type from both neighbor and non-neighbor groups
        # to only consider inter-cell-type interactions
        if ct in motif:
            center_cell_mask = self.labels[neighbor_cells] == ct
            neighbor_cells_filtered = neighbor_cells[~center_cell_mask]

            center_cell_mask_non = self.labels[non_neighbor_cells] == ct
            non_neighbor_cells_filtered = non_neighbor_cells[~center_cell_mask_non]

            n_removed_neighbor = len(neighbor_cells) - len(neighbor_cells_filtered)
            n_removed_non_neighbor = len(non_neighbor_cells) - len(non_neighbor_cells_filtered)
            
            print(f'Focus on inter-cell-type interactions: Remove {n_removed_neighbor} center type cells from neighbor group, {n_removed_non_neighbor} from non-neighbor groups.')

            # Use filtered cell lists
            neighbor_cells = neighbor_cells_filtered
            non_neighbor_cells = non_neighbor_cells_filtered

        if len(neighbor_cells) < 3:
            raise ValueError(f"Not enough neighbor cells ({len(neighbor_cells)}) for correlation analysis. Need at least 3 cells.")
        if len(non_neighbor_cells) < 3:
            raise ValueError(f"Not enough non-neighbor cells ({len(non_neighbor_cells)}) for correlation analysis. Need at least 3 cells.")

        print(f"Find {len(neighbor_result['center_id'])} center cells with motif nearby.")
        print(f'Find {len(neighbor_cells)} neighbor cells and {len(non_neighbor_cells)} non-neighbor cells in motif')

        # Get gene list
        if genes is None:
            genes = self.genes
        elif isinstance(genes, str):
            genes = [genes]

        # Filter genes that exist
        valid_genes = [g for g in genes if g in self.genes]
        if len(valid_genes) == 0:
            raise ValueError("No valid genes found in the dataset.")

        # Get center cells for cross-correlation with nearby motif cells
        center_cells = neighbor_result['center_id']

        # Get all cells of each type for computing cell type means
        center_type_mask = self.labels == ct
        all_center_cells = np.where(center_type_mask)[0]

        motif_mask = np.isin(np.array(self.labels), motif)
        motif_mask_filtered = motif_mask & (~center_type_mask)
        all_motif_cells = np.where(motif_mask_filtered)[0]

        if len(all_motif_cells) == 0:
            raise ValueError(f"No motif cells found after excluding center cell type '{ct}'. "
                           f"Cannot compute cell type mean for shifted correlation.")

        print(f"Computing cross-correlation between center cells ({ct}) and motif cells")
        print(f"Center cells for correlation: {len(center_cells)}")
        print(f"Neighbor motif cells: {len(neighbor_cells)}")
        print(f"Non-neighbor motif cells: {len(non_neighbor_cells)}")

        # Extract expression data (keep as sparse if possible)
        center_expr = self.adata[center_cells, valid_genes].X
        neighbor_expr = self.adata[neighbor_cells, valid_genes].X
        non_neighbor_expr = self.adata[non_neighbor_cells, valid_genes].X

        # Get all cells for computing means
        all_center_expr = self.adata[all_center_cells, valid_genes].X
        all_motif_expr = self.adata[all_motif_cells, valid_genes].X

        # Check if sparse
        is_sparse = sparse.issparse(neighbor_expr)

        # Filter genes by non-zero expression (work with sparse matrix)
        if is_sparse:
            nonzero_center = np.array((center_expr > 0).sum(axis=0)).flatten()
            nonzero_neighbor = np.array((neighbor_expr > 0).sum(axis=0)).flatten()
            nonzero_non_neighbor = np.array((non_neighbor_expr > 0).sum(axis=0)).flatten()
        else:
            nonzero_center = (center_expr > 0).sum(axis=0)
            nonzero_neighbor = (neighbor_expr > 0).sum(axis=0)
            nonzero_non_neighbor = (non_neighbor_expr > 0).sum(axis=0)

        valid_gene_mask = (nonzero_center >= min_nonzero) & (nonzero_neighbor >= min_nonzero) & (nonzero_non_neighbor >= min_nonzero)

        if valid_gene_mask.sum() == 0:
            raise ValueError(f"No genes passed the min_nonzero={min_nonzero} filter.")

        # Apply gene filter
        center_expr = center_expr[:, valid_gene_mask]
        neighbor_expr = neighbor_expr[:, valid_gene_mask]
        non_neighbor_expr = non_neighbor_expr[:, valid_gene_mask]
        all_center_expr = all_center_expr[:, valid_gene_mask]
        all_motif_expr = all_motif_expr[:, valid_gene_mask]
        filtered_genes = [valid_genes[i] for i in range(len(valid_genes)) if valid_gene_mask[i]]

        print(f"After filtering (min_nonzero={min_nonzero}): {len(filtered_genes)} genes")

        # Compute cell type means (keep sparse)
        from time import time
        start_time = time()
        if is_sparse:
            center_mean = np.array(all_center_expr.mean(axis=0)).flatten()  
            motif_mean = np.array(all_motif_expr.mean(axis=0)).flatten()
            
            sum_center = np.array(center_expr.sum(axis=0)).flatten()
            sum_neighbor = np.array(neighbor_expr.sum(axis=0)).flatten()
            sum_non_neighbor = np.array(non_neighbor_expr.sum(axis=0)).flatten()
            
            sum_sq_center = np.array(center_expr.power(2).sum(axis=0)).flatten()
            sum_sq_neighbor = np.array(neighbor_expr.power(2).sum(axis=0)).flatten()
            sum_sq_non_neighbor = np.array(non_neighbor_expr.power(2).sum(axis=0)).flatten()
        else:
            center_mean = all_center_expr.mean(axis=0)
            motif_mean = all_motif_expr.mean(axis=0)
            
            sum_center = center_expr.sum(axis=0)
            sum_neighbor = neighbor_expr.sum(axis=0)
            sum_non_neighbor = non_neighbor_expr.sum(axis=0)
            
            sum_sq_center = (center_expr**2).sum(axis=0)
            sum_sq_neighbor = (neighbor_expr**2).sum(axis=0)
            sum_sq_non_neighbor = (non_neighbor_expr**2).sum(axis=0)
        
        n_center = len(center_cells)
        n_neighbor = len(neighbor_cells)
        n_non_neighbor = len(non_neighbor_cells)
        
        # Compute cross-covariance using expanded formula
        # Cov = [∑∑(x*y) - n_neighbor*μ_motif*∑x - n_center*μ_center*∑y + n_center*n_neighbor*μ_center*μ_motif] / (n_center * n_neighbor)
        
        sum_product_neighbor = np.outer(sum_center, sum_neighbor)
        sum_product_non_neighbor = np.outer(sum_center, sum_non_neighbor)
        
        mean_outer = np.outer(center_mean, motif_mean)
        
        cross_cov_neighbor = (
            sum_product_neighbor / (n_center * n_neighbor)
            - np.outer(center_mean, sum_neighbor / n_neighbor)
            - np.outer(sum_center / n_center, motif_mean)
            + mean_outer
        )
        
        cross_cov_non_neighbor = (
            sum_product_non_neighbor / (n_center * n_non_neighbor)
            - np.outer(center_mean, sum_non_neighbor / n_non_neighbor)
            - np.outer(sum_center / n_center, motif_mean)
            + mean_outer
        )
        
        # Compute variances
        # Var = E[(X - μ_all)²] = E[X²] - 2*μ_all*E[X] + μ_all²
        var_center = (sum_sq_center / n_center 
                    - 2 * center_mean * sum_center / n_center 
                    + center_mean**2)
        
        var_neighbor = (sum_sq_neighbor / n_neighbor
                    - 2 * motif_mean * sum_neighbor / n_neighbor
                    + motif_mean**2)
        
        var_non_neighbor = (sum_sq_non_neighbor / n_non_neighbor
                        - 2 * motif_mean * sum_non_neighbor / n_non_neighbor
                        + motif_mean**2)
        
        # Standard deviations
        std_center = np.sqrt(np.maximum(var_center, 0))
        std_neighbor = np.sqrt(np.maximum(var_neighbor, 0))
        std_non_neighbor = np.sqrt(np.maximum(var_non_neighbor, 0))
        
        # Correlation matrices
        std_outer_neighbor = np.outer(std_center, std_neighbor)
        std_outer_neighbor[std_outer_neighbor == 0] = 1e-10
        
        std_outer_non_neighbor = np.outer(std_center, std_non_neighbor)
        std_outer_non_neighbor[std_outer_non_neighbor == 0] = 1e-10
        
        corr_matrix_neighbor = cross_cov_neighbor / std_outer_neighbor
        corr_matrix_non_neighbor = cross_cov_non_neighbor / std_outer_non_neighbor

        end_time = time()

        print(f'Time for computing cross correlation: {end_time-start_time:.2f} seconds')

        ### Now do the same analysis but for center cells without motif nearby
        # Center: ct cells WITHOUT motif nearby (all ct cells - center_id from get_motif_neighbor_cells)
        # Neighbor: ONLY cells around these "non-motif" centers (excluding center type cells)

        print("\n" + "="*60)
        print("Computing correlation for center cells WITHOUT motif nearby")
        print("="*60)

        # Get neighbors for centers WITHOUT motif by excluding centers with motif
        no_motif_result = self.get_all_neighbor_cells(
            ct=ct,
            max_dist=max_dist,
            k=k,
            min_size=min_size,
            exclude_centers=neighbor_result['center_id']  # Exclude centers with motif
        )

        centers_without_motif = no_motif_result['center_id']
        centers_without_motif_neighbors = no_motif_result['neighbor_id']

        # Remove overlap between neighbor_cells (with motif) and centers_without_motif_neighbors
        # Cells that are neighbors of both center_with_motif and center_without_motif
        # should be attributed to the center_with_motif group (i.e., neighbor_cells)
        overlap_neighbors = np.intersect1d(neighbor_cells, centers_without_motif_neighbors)
        if len(overlap_neighbors) > 0:
            print(f"Found {len(overlap_neighbors)} cells that are neighbors of both center_with_motif and center_without_motif")
            print(f"These cells will be kept in the neighbor_with_motif group and removed from centers_without_motif_neighbors")
            centers_without_motif_neighbors = np.setdiff1d(centers_without_motif_neighbors, overlap_neighbors)

        if len(centers_without_motif) == 0:
            print("No center cells without motif nearby. Skipping this analysis.")
        elif len(centers_without_motif_neighbors) < 3:
            print(f"Not enough neighbors ({len(centers_without_motif_neighbors)}) for centers without motif. Skipping.")
            centers_without_motif = np.array([])
        else:
            print(f"Found {len(centers_without_motif)} center cells without motif nearby")
            print(f"Found {len(centers_without_motif_neighbors)} neighbors (excluding center type cells and overlap with neighbor_with_motif)")

        # Only compute if we have enough data
        if len(centers_without_motif) > 10 and len(centers_without_motif_neighbors) >= 10:
            print(f"Neighbors of centers without motif: {len(centers_without_motif_neighbors)}")

            # Extract expression data for this new group
            center_no_motif_expr = self.adata[centers_without_motif, valid_genes].X[:, valid_gene_mask]
            neighbor_no_motif_expr = self.adata[centers_without_motif_neighbors, valid_genes].X[:, valid_gene_mask]

            # Compute statistics (keep sparse)
            if is_sparse:
                sum_center_no_motif = np.array(center_no_motif_expr.sum(axis=0)).flatten()
                sum_neighbor_no_motif = np.array(neighbor_no_motif_expr.sum(axis=0)).flatten()
                sum_sq_center_no_motif = np.array(center_no_motif_expr.power(2).sum(axis=0)).flatten()
                sum_sq_neighbor_no_motif = np.array(neighbor_no_motif_expr.power(2).sum(axis=0)).flatten()
            else:
                sum_center_no_motif = center_no_motif_expr.sum(axis=0)
                sum_neighbor_no_motif = neighbor_no_motif_expr.sum(axis=0)
                sum_sq_center_no_motif = (center_no_motif_expr**2).sum(axis=0)
                sum_sq_neighbor_no_motif = (neighbor_no_motif_expr**2).sum(axis=0)

            n_center_no_motif = len(centers_without_motif)
            n_neighbor_no_motif = len(centers_without_motif_neighbors)

            # Compute cross-covariance
            sum_product_no_motif = np.outer(sum_center_no_motif, sum_neighbor_no_motif)

            cross_cov_no_motif = (
                sum_product_no_motif / (n_center_no_motif * n_neighbor_no_motif)
                - np.outer(center_mean, sum_neighbor_no_motif / n_neighbor_no_motif)
                - np.outer(sum_center_no_motif / n_center_no_motif, motif_mean)
                + mean_outer
            )

            # Compute variance for neighbors (shifted with the same mean)
            var_neighbor_no_motif = (
                sum_sq_neighbor_no_motif / n_neighbor_no_motif
                - 2 * motif_mean * sum_neighbor_no_motif / n_neighbor_no_motif
                + motif_mean**2
            )
            var_center_no_motif = (
                sum_sq_center_no_motif / n_center_no_motif
                - 2 * center_mean * sum_center_no_motif / n_center_no_motif
                + center_mean**2
            )

            std_neighbor_no_motif = np.sqrt(np.maximum(var_neighbor_no_motif, 0))
            std_center_no_motif = np.sqrt(np.maximum(var_center_no_motif, 0))

            # Correlation matrix
            std_outer_no_motif = np.outer(std_center_no_motif, std_neighbor_no_motif)
            std_outer_no_motif[std_outer_no_motif == 0] = 1e-10

            corr_matrix_no_motif = cross_cov_no_motif / std_outer_no_motif

            print(f"Computed cross-correlation for centers without motif")
        else:
            print("Not enough data for centers without motif. Setting correlation matrix to None.")
            corr_matrix_no_motif = None

        # Prepare results - combine all three correlations and perform statistical tests
        print("\n" + "="*60)
        print("Performing Fisher Z-tests and preparing results")
        print("="*60)

        from scipy import stats as scipy_stats
        from statsmodels.stats.multitest import multipletests

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

        # Effective sample sizes for cross-correlation
        # For cross-correlation between n_center and n_neighbor samples,
        # the effective sample size is approximately min(n_center, n_neighbor)
        # This is more conservative than n_center * n_neighbor
        # n_eff_neighbor = min(n_center, n_neighbor)
        # n_eff_non_neighbor = min(n_center, n_non_neighbor)
        n_eff_neighbor = n_center + n_neighbor - 1
        n_eff_non_neighbor = n_center + n_non_neighbor - 1
    

        # Test 1: Corr1 vs Corr2 (neighbor vs non-neighbor) - vectorized
        z_stat_test1, p_value_test1 = fisher_z_test_vectorized(
            corr_matrix_neighbor, n_eff_neighbor,
            corr_matrix_non_neighbor, n_eff_non_neighbor
        )
        delta_corr_test1 = corr_matrix_neighbor - corr_matrix_non_neighbor

        # Test 2: Corr1 vs Corr3 (neighbor vs no_motif) - vectorized if available
        if corr_matrix_no_motif is not None:
            n_eff_no_motif = min(n_center_no_motif, n_neighbor_no_motif)
            z_stat_test2, p_value_test2 = fisher_z_test_vectorized(
                corr_matrix_neighbor, n_eff_neighbor,
                corr_matrix_no_motif, n_eff_no_motif
            )
            delta_corr_test2 = corr_matrix_neighbor - corr_matrix_no_motif

            # Combined score (vectorized)
            combined_score = (0.4 * delta_corr_test1 * (-np.log10(p_value_test1 + 1e-300)) +
                            0.6 * delta_corr_test2 * (-np.log10(p_value_test2 + 1e-300)))
        else:
            z_stat_test2 = None
            p_value_test2 = None
            delta_corr_test2 = None
            combined_score = None

        # Create meshgrid for gene pairs (vectorized)
        n_genes = len(filtered_genes)
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

        # Apply sequential filtering and FDR correction
        print(f"\nTotal gene pairs: {len(results_df)}")

        if corr_matrix_no_motif is not None:
            # Step 1: Filter by p1 < 0.05
            mask_test1 = results_df['p_value_test1'] < 0.05
            print(f"Gene pairs passing test1 (p < 0.05): {mask_test1.sum()}")

            # Step 2: Filter by p2 < 0.05
            mask_test2 = results_df['p_value_test2'] < 0.05
            print(f"Gene pairs passing test2 (p < 0.05): {mask_test2.sum()}")

            # Step 3: Check direction consistency
            # Both delta_corr should have the same sign (both positive or both negative)
            same_direction = np.sign(results_df['delta_corr_test1']) == np.sign(results_df['delta_corr_test2'])
            

            # Intersection: both tests significant AND same direction
            mask_both = mask_test1 & mask_test2 & same_direction
            print(f"Gene pairs passing both tests with consistent direction: {mask_both.sum()}")

            # FDR correction using Benjamini-Hochberg method for candidates passing both filters
            if mask_both.sum() > 0:
                # FDR for test1 p-values (among candidates)
                p_values_test1_candidates = results_df.loc[mask_both, 'p_value_test1'].values
                reject_test1, q_values_test1, _, _ = multipletests(
                    p_values_test1_candidates,
                    alpha=0.05,
                    method='fdr_bh'
                )
                results_df.loc[mask_both, 'q_value_test1'] = q_values_test1
                results_df.loc[mask_both, 'reject_test1_fdr'] = reject_test1

                # FDR for test2 p-values (among candidates)
                p_values_test2_candidates = results_df.loc[mask_both, 'p_value_test2'].values
                reject_test2, q_values_test2, _, _ = multipletests(
                    p_values_test2_candidates,
                    alpha=0.05,
                    method='fdr_bh'
                )
                results_df.loc[mask_both, 'q_value_test2'] = q_values_test2
                results_df.loc[mask_both, 'reject_test2_fdr'] = reject_test2

                print(f"FDR correction completed for {mask_both.sum()} candidates")
                print(f"  - Test1 FDR significant (q < 0.05): {reject_test1.sum()}")
                print(f"  - Test2 FDR significant (q < 0.05): {reject_test2.sum()}")
            else:
                results_df['q_value_test1'] = np.nan
                results_df['q_value_test2'] = np.nan
                results_df['reject_test1_fdr'] = False
                results_df['reject_test2_fdr'] = False
        else:
            # Only test1 available
            print("Note: Test2 not available (no centers without motif)")
            results_df['q_value_test1'] = np.nan
            results_df['q_value_test2'] = np.nan
            results_df['reject_test1_fdr'] = False
            results_df['reject_test2_fdr'] = False

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

        # Prepare cell groups dictionary
        cell_groups = {
            'center_with_motif': center_cells,
            'center_without_motif': centers_without_motif if len(centers_without_motif) > 0 else np.array([]),
            'neighbor_motif': neighbor_cells,
            'non_neighbor_motif': non_neighbor_cells,
            'center_without_motif_neighbors': centers_without_motif_neighbors if len(centers_without_motif) > 0 else np.array([])
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
            fig_size: tuple = (6, 6),
            save_path: Optional[str] = None,
            ):
        """
        Plot the cell type distribution of single fov.

        Parameter
        --------
        center:
            Center cell type.
        ids: dict. Output of spatial_query.get_motif_neighbor_cells
            A dictionary containing cell indices for different groups:
                - 'center_with_motif': center cells with motif nearby
                - 'center_without_motif': center cells without motif nearby
                - 'neighbor_motif': motif cells that are neighbors of center cells
                - 'non_neighbor_motif': motif cells that are not neighbors of center cells
                - 'center_without_motif_neighbors': neighbors of center cells without motif
                  (excluding any overlap with neighbor_motif - overlapping cells are attributed
                  to the neighbor_motif group)
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
        adata.obs.iloc[ids['center_with_motif'], adata.obs.columns.get_loc('tmp')] = f'center {center} with motif'
        adata.obs.iloc[ids['center_without_motif'], adata.obs.columns.get_loc('tmp')] = f'non-motif center {center}'

        adata.obs.iloc[ids['neighbor_motif'], adata.obs.columns.get_loc('tmp')] = [f'neighbor motif: {self.labels[i]}' for i in ids['neighbor_motif']]
        adata.obs.iloc[ids['non_neighbor_motif'], adata.obs.columns.get_loc('tmp')] = [f'non-neighbor motif: {self.labels[i]}' for i in ids['non_neighbor_motif']]

        adata.obs.iloc[ids['center_without_motif_neighbors'], adata.obs.columns.get_loc('tmp')] = 'non-motif-center neighbors'

        neighbor_motif_types = adata.obs.iloc[ids['neighbor_motif'], adata.obs.columns.get_loc(self.label_key)].unique()
        non_neighbor_motif_types = adata.obs.iloc[ids['non_neighbor_motif'], adata.obs.columns.get_loc(self.label_key)].unique()

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
        fig, ax = plt.subplots(figsize=(6, 6))
        sc.pl.embedding(adata, basis='X_spatial', color='tmp', palette=color_dict, size=10, ax=ax, show=False)
        ax.set_title(f'Cell types around {center} with motif', fontsize=10)
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()