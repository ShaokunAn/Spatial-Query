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
from scipy.stats import hypergeom, fisher_exact
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests
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
        # Store spatial position and cell type label
        self.spatial_key = spatial_key
        self.spatial_pos = np.array(adata.obsm[self.spatial_key])
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
        var_df = adata.var.reset_index()  # 确保有 index 信息
        duplicated = var_df.duplicated(subset=[feature_name], keep='first')
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
            plt.tight_layout(rect=[0, 0, 1.1, 1])
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

    def plot_motif_celltype(self,
                            ct: str,
                            motif: Union[str, List[str]],
                            max_dist: float = 100,
                            fig_size: tuple = (10, 5),
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
        plt.tight_layout(rect=[0, 0, 1.1, 1])
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
