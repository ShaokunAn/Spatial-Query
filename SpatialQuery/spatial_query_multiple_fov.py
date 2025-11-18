from typing import List, Union, Optional, Dict, Tuple, Any, Literal

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multitest as mt
from anndata import AnnData
from mlxtend.frequent_patterns import fpgrowth
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.stats import hypergeom

import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests

from scipy import sparse
from . import spatial_utils

from .spatial_query import spatial_query
from .spatial_utils import (
    find_maximal_patterns,
    query_pattern,
    build_fptree_knn,
    build_fptree_dist,
    de_genes_scanpy,
    de_genes_fisher,
    )
import anndata as ad

from time import time


class spatial_query_multi:
    def __init__(
        self,
        adatas: List[AnnData],
        datasets: List[str],
        spatial_key: str,
        label_key: str,
        leaf_size: int = 10,
        max_radius: float = 500,
        n_split: int = 10,
        build_gene_index: bool = False,
        feature_name: str = None,
        if_lognorm: bool = True,
        ):
        """
        Initiate models, including setting attributes and building kd-tree for each field of view.

        Parameter
        ---------
        adatas:
            List of adata
        datasets:
            List of dataset names
        spatial_key:
            Spatial coordination name in AnnData.obsm object
        label_key:
            Label name in AnnData.obs object
        leaf_size:
            The largest number of points stored in each leaf node.
        max_radius: 
            The upper limit of neighborhood radius.
        n_split:
            The number of splits in each axis for spatial grid to speed up query, default is 10
        build_gene_index:
            Whether to build scfind index or use adata.X directly. If set True, build scfind index otherwise use adata.X directly.
        feature_name:
            The label or key in the AnnData object's variables (var) that corresponds to the feature names. 
        if_lognorm:
            Whether to log normalize the expression data, default is True
        """
        # Each element in self.spatial_queries stores a spatial_query object
        self.spatial_key = spatial_key
        self.label_key = label_key
        self.max_radius = max_radius
        self.build_gene_index = build_gene_index

        # Modify dataset names by d_0, d_2, ... for duplicates in datasets
        count_dict = {}
        modified_datasets = []
        for dataset in datasets:
            if '_' in dataset:
                print(f"Replacing _ with hyphen in {dataset}.")
                dataset = dataset.replace('_', '-')

            if dataset in count_dict:
                count_dict[dataset] += 1
            else:
                count_dict[dataset] = 0

            mod_dataset = f"{dataset}_{count_dict[dataset]}"
            modified_datasets.append(mod_dataset)

        self.datasets = modified_datasets

        self.spatial_queries = [spatial_query(
            adata=adata, 
            dataset=self.datasets[i],
            spatial_key=spatial_key,
            label_key=label_key,
            leaf_size=leaf_size,
            max_radius=self.max_radius,
            n_split=n_split,
            build_gene_index=build_gene_index,
            feature_name=feature_name,
            if_lognorm=if_lognorm,
            ) for i, adata in enumerate(adatas)]
        

    def find_fp_knn(self,
                    ct: str,
                    dataset: Union[str, List[str]] = None,
                    k: int = 30,
                    min_support: float = 0.5,
                    max_dist: float = 500
                    ) -> pd.DataFrame:
        """
        Find frequent patterns within the KNNs of certain cell type in multiple fields of view.

        Parameter
        ---------
        ct:
            Cell type name.
        dataset:
            Datasets for searching for frequent patterns.
            Use all datasets if dataset=None.
        k:
            Number of nearest neighbors.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        max_dist:
            The maximum distance at which points are considered neighbors.

        Return
        ------
        Frequent patterns in the neighborhood of certain cell type.
        """
        # Search transactions for each field of view, find the frequent patterns of integrated transactions
        # start = time.time()
        if_exist_label = [ct in s.labels.unique() for s in self.spatial_queries]
        if not any(if_exist_label):
            raise ValueError(f"Found no {self.label_key} in all datasets!")

        if dataset is None:
            # Use all datasets if dataset is not provided
            dataset = [s.dataset.split('_')[0] for s in self.spatial_queries]

        # Make sure dataset is a list
        if isinstance(dataset, str):
            dataset = [dataset]

        # test if the input dataset name is valid
        valid_ds_names = [s.dataset.split('_')[0] for s in self.spatial_queries]
        for ds in dataset:
            if ds not in valid_ds_names:
                raise ValueError(f"Invalid input dataset name: {ds}.\n "
                                 f"Valid dataset names are: {set(valid_ds_names)}")

        max_dist = min(max_dist, self.max_radius)
        # end = time.time()
        # print(f"time for checking validation of inputs: {end-start} seconds")

        # start = time.time()
        transactions = []
        for s in self.spatial_queries:
            if s.dataset.split('_')[0] not in dataset:
                continue
            cell_pos = s.spatial_pos
            labels = np.array(s.labels)
            if ct not in np.unique(labels):
                continue

            ct_pos = cell_pos[labels == ct]
            dists, idxs = s.kd_tree.query(ct_pos, k=k + 1, workers=-1)
            mask = dists < max_dist
            for i, idx in enumerate(idxs):
                inds = idx[mask[i]]
                transaction = labels[inds[1:]]
                transactions.append(transaction)
        # end = time.time()
        # print(f"time for building {len(transactions)} transactions: {end-start} seconds.")

        # start = time.time()
        mlb = MultiLabelBinarizer()
        encoded_data = mlb.fit_transform(transactions)
        df = pd.DataFrame(encoded_data.astype(bool), columns=mlb.classes_)
        # end = time.time()
        # print(f"time for building df for fpgrowth: {end-start} seconds")

        # start = time.time()
        fp = fpgrowth(df, min_support=min_support, use_colnames=True)
        if len(fp) == 0:
            return pd.DataFrame(columns=['support', 'itemsets'])
        # end = time.time()
        # print(f"time for find fp_growth: {end-start} seconds, {len(fp)} frequent patterns.")
        # start = time.time()
        fp = find_maximal_patterns(fp=fp)
        # end = time.time()
        # print(f"time for identify maximal patterns: {end - start} seconds")

        fp['itemsets'] = fp['itemsets'].apply(lambda x: list(sorted(x)))
        fp.sort_values(by='support', ascending=False, inplace=True, ignore_index=True)

        return fp

    def find_fp_dist(self,
                     ct: str,
                     dataset: Union[str, List[str]] = None,
                     max_dist: float = 100,
                     min_size: int = 0,
                     min_support: float = 0.5,
                     max_ns: int = 100
                     ):
        """
        Find frequent patterns within the radius of certain cell type in multiple fields of view.

        Parameter
        ---------
        ct:
            Cell type name.
        dataset:
            Datasets for searching for frequent patterns.
            Use all datasets if dataset=None.
        max_dist:
            Maximum distance for considering a cell as a neighbor.
        min_size:
            Minimum neighborhood size for each point to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        max_ns:
            Upper limit of neighbors for each point.

        Return
        ------
        Frequent patterns in the neighborhood of certain cell type.
        """
        # Search transactions for each field of view, find the frequent patterns of integrated transactions
        if_exist_label = [ct in s.labels.unique() for s in self.spatial_queries]
        if not any(if_exist_label):
            raise ValueError(f"Found no {self.label_key} in any datasets!")

        if dataset is None:
            dataset = [s.dataset.split('_')[0] for s in self.spatial_queries]
        if isinstance(dataset, str):
            dataset = [dataset]

        # test if the input dataset name is valid
        valid_ds_names = [s.dataset.split('_')[0] for s in self.spatial_queries]
        for ds in dataset:
            if ds not in valid_ds_names:
                raise ValueError(f"Invalid input dataset name: {ds}.\n "
                                 f"Valid dataset names are: {set(valid_ds_names)}")

        max_dist = min(max_dist, self.max_radius)
        # start = time.time()
        transactions = []
        for s in self.spatial_queries:
            if s.dataset.split('_')[0] not in dataset:
                continue
            cell_pos = s.spatial_pos
            labels = np.array(s.labels)
            if ct not in np.unique(labels):
                continue

            cinds = [id for id, l in enumerate(labels) if l == ct]
            ct_pos = cell_pos[cinds]

            idxs = s.kd_tree.query_ball_point(ct_pos, r=max_dist, return_sorted=False, workers=-1)

            for i_id, idx in zip(cinds, idxs):
                transaction = [labels[i] for i in idx[:min(max_ns, len(idx))] if i != i_id]
                if len(transaction) > min_size:
                    transactions.append(transaction)

        # end = time.time()
        # print(f"time for building {len(transactions)} transactions: {end-start} seconds.")

        # start = time.time()
        mlb = MultiLabelBinarizer()
        encoded_data = mlb.fit_transform(transactions)
        df = pd.DataFrame(encoded_data.astype(bool), columns=mlb.classes_)
        # end = time.time()
        # print(f"time for building df for fpgrowth: {end - start} seconds")

        # start = time.time()
        fp = fpgrowth(df, min_support=min_support, use_colnames=True)
        if len(fp) == 0:
            return pd.DataFrame(columns=['support', 'itemsets'])
        # end = time.time()
        # print(f"time for find fp_growth: {end - start} seconds, {len(fp)} frequent patterns.")

        # start = time.time()
        fp = find_maximal_patterns(fp=fp)
        # end = time.time()
        # print(f"time for identify maximal patterns: {end - start} seconds")

        fp['itemsets'] = fp['itemsets'].apply(lambda x: list(sorted(x)))
        fp.sort_values(by='support', ascending=False, inplace=True, ignore_index=True)

        return fp

    def motif_enrichment_knn(self,
                             ct: str,
                             motifs: Union[str, List[str]] = None,
                             dataset: Union[str, List[str]] = None,
                             k: int = 30,
                             min_support: float = 0.5,
                             max_dist: float = 500.0,
                             return_cellID: bool = False,
                             ) -> pd.DataFrame:
        """
        Perform motif enrichment analysis using k-nearest neighbors (KNN) in multiple fields of view.

        Parameter
        ---------
        ct:
            The cell type of the center cell.
        motifs:
            Specified motifs to be tested.
            If motifs=None, find the frequent patterns as motifs within
            the neighborhood of center cell type in each fov.
        dataset:
            Datasets for searching for frequent patterns and performing enrichment analysis.
            Use all datasets if dataset=None.
        k:
            Number of nearest neighbors to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        dis_duplicates:
            Distinguish duplicates in patterns if dis_duplicates=True. This will consider transactions within duplicates
            like (A, A, A, B, C) otherwise only patterns with unique cell types will be considered like (A, B, C).
        max_dist:
            Maximum distance for neighbors (default: 500).
        return_cellID:
            Indicate whether return cell IDs for each frequent pattern within the neighborhood of center cell type and center cells.
            By defaults do not return cell ID.

        Return
        ------
        If return_cellID is False:
            pd.DataFrame containing statistical measures for motif enrichment.
        If return_cellID is True:
            A tuple with three elements:
            - The original DataFrame output
            - Dictionary with cell IDs of motifs in center cell's neighborhood in each dataset for each motif:
              {'motif_1': {'dataset_1': [ids]}}
            - Dictionary with cell IDs of center cell type with given motif in their neighborhood:
              {'motif_1': {'dataset_1': [ids]}}
        """
        if dataset is None:
            dataset = [s.dataset.split('_')[0] for s in self.spatial_queries]
        if isinstance(dataset, str):
            dataset = [dataset]

        max_dist = min(max_dist, self.max_radius)

        out = []
        if_exist_label = [ct in s.labels.unique() for s in self.spatial_queries]
        if not any(if_exist_label):
            raise ValueError(f"Found no {self.label_key} in any datasets!")

        # Check whether specify motifs. If not, search frequent patterns among specified datasets
        # and use them as interested motifs
        all_labels = pd.concat([s.labels for s in self.spatial_queries])
        labels_unique_all = set(all_labels.unique())
        if motifs is None:
            fp = self.find_fp_knn(ct=ct, k=k, dataset=dataset,
                                  min_support=min_support, max_dist=max_dist)
            motifs = fp['itemsets'].tolist()
        else:
            if isinstance(motifs, str):
                motifs = [motifs]

            motifs_exc = [m for m in motifs if m not in labels_unique_all]
            if len(motifs_exc) != 0:
                print(f"Found no {motifs_exc} in {dataset}. Ignoring them.")
            motifs = [m for m in motifs if m not in motifs_exc]
            if len(motifs) == 0:
                raise ValueError(f"All cell types in motifs are missed in {self.label_key}.")
            motifs = [motifs]

        # Initialize dictionaries to store cell IDs if requested
        motif_cell_ids = {}
        center_cell_ids = {}

        for motif in motifs:
            n_labels = 0
            n_ct = 0
            n_motif_labels = 0
            n_motif_ct = 0

            motif = list(motif) if not isinstance(motif, list) else motif
            sort_motif = sorted(motif)

            if return_cellID:
                motif_str = str(sort_motif)
                motif_cell_ids[motif_str] = {}
                center_cell_ids[motif_str] = {}

            # Calculate statistics of each dataset
            for fov, s in enumerate(self.spatial_queries):
                if s.dataset.split('_')[0] not in dataset:
                    continue

                cell_pos = s.spatial_pos
                labels = np.array(s.labels)
                labels_unique = np.unique(labels)

                contain_motif = [m in labels_unique for m in motif]
                if not np.all(contain_motif):
                    n_labels += labels.shape[0]
                    n_ct += np.sum(labels == ct)
                    continue
                else:
                    n_labels += labels.shape[0]
                    label_encoder = LabelEncoder()
                    int_labels = label_encoder.fit_transform(labels)
                    int_motifs = label_encoder.transform(np.array(motif))

                    dists, idxs = s.kd_tree.query(cell_pos, k=k + 1, workers=-1)
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

                    # Check which cells have all motif types in their neighborhood
                    motif_mask = np.all(neighbor_counts[:, int_motifs] > 0, axis=1)
                    n_motif_labels += np.sum(motif_mask)

                    if ct in np.unique(labels):
                        int_ct = label_encoder.transform(np.array(ct, dtype=object, ndmin=1))
                        mask = int_labels == int_ct
                        center_mask = mask & motif_mask
                        n_motif_ct += np.sum(center_mask)
                        n_ct += np.sum(mask)

                        if return_cellID:
                            # Get IDs of center cells with motif in neighborhood
                            center_indices = np.where(center_mask)[0]

                            if len(center_indices) > 0:
                                # Get all neighbors of center cells that have the motif
                                center_neighbors_idxs = filtered_idxs[center_mask]

                                # Flatten and get unique neighbors
                                all_neighbors_center = center_neighbors_idxs.flatten()
                                all_neighbors_center = all_neighbors_center[all_neighbors_center != -1]

                                # Filter to only keep neighbors that are of motif cell types
                                motif_mask_all = np.isin(np.array(s.labels), motif)
                                valid_neighbors_center = all_neighbors_center[motif_mask_all[all_neighbors_center]]
                                id_motif_celltype = set(valid_neighbors_center)

                                motif_cell_ids[motif_str][s.dataset] = list(id_motif_celltype)
                                center_cell_ids[motif_str][s.dataset] = list(center_indices)

            if ct in motif:
                n_ct = round(n_ct / motif.count(ct))

            hyge = hypergeom(M=n_labels, n=n_ct, N=n_motif_labels)
            motif_out = {'center': ct, 'motifs': sort_motif, 'n_center_motif': n_motif_ct,
                         'n_center': n_ct, 'n_motif': n_motif_labels, 'expectation': hyge.mean(), 'p-values': hyge.sf(n_motif_ct)}
            out.append(motif_out)

        out_pd = pd.DataFrame(out)

        if len(out_pd) == 1:
            out_pd['if_significant'] = True if out_pd['p-values'][0] < 0.05 else False
        else:
            p_values = out_pd['p-values'].tolist()
            if_rejected, corrected_p_values = mt.fdrcorrection(p_values,
                                                               alpha=0.05,
                                                               method='poscorr')
            out_pd['adj_pvals'] = corrected_p_values
            out_pd['if_significant'] = if_rejected
            out_pd = out_pd.sort_values(by='adj_pvals', ignore_index=True)

        if return_cellID:
            return out_pd, motif_cell_ids, center_cell_ids
        else:
            return out_pd

    def motif_enrichment_dist(self,
                              ct: str,
                              motifs: Union[str, List[str], List[List[str]]] = None,
                              dataset: Union[str, List[str]] = None,
                              max_dist: float = 100,
                              min_size: int = 0,
                              min_support: float = 0.5,
                              max_ns: int = 100,
                              return_cellID: bool = False
                              ):
        """
        Perform motif enrichment analysis within a specified radius-based neighborhood in multiple fields of view.

        Parameter
        ---------
        ct:
            Cell type of the center cell.
        motifs:
            Specified motifs to be tested.
            If motifs=None, find the frequent patterns as motifs within the neighborhood of center cell type.
        dataset:
            Datasets for searching for frequent patterns and performing enrichment analysis.
            Use all datasets if dataset=None.
        max_dist:
            Maximum distance for considering a cell as a neighbor.
        min_size:
            Minimum neighborhood size for each point to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        dis_duplicates:
            Distinguish duplicates in patterns if dis_duplicates=True. This will consider transactions within duplicates
            like (A, A, A, B, C) otherwise only patterns with unique cell types will be considered like (A, B, C).
        max_ns:
            Maximum number of neighborhood size for each point.
        return_cellID:
            Indicate whether return cell IDs for each frequent pattern within the neighborhood of grid points.
            By defaults do not return cell ID.
        Returns
        -------
        If return_cellID is False:
            pd.DataFrame containing statistical measures for motif enrichment.
        If return_cellID is True:
            A tuple with three elements:
            - The original DataFrame output
            - Dictionary with cell IDs of motifs in center cell's neighborhood in each dataset for each motif:
              {'motif_1': {'dataset_1': [ids]}}
            - Dictionary with cell IDs of center cell type with given motif in their neighborhood:
              {'motif_1': {'dataset_1': [ids]}}
        """
        if dataset is None:
            dataset = [s.dataset.split('_')[0] for s in self.spatial_queries]
        if isinstance(dataset, str):
            dataset = [dataset]

        out = []
        if_exist_label = [ct in s.labels.unique() for s in self.spatial_queries]
        if not any(if_exist_label):
            raise ValueError(f"Found no {self.label_key} in any datasets!")

        max_dist = min(max_dist, self.max_radius)

        # Check whether specify motifs. If not, search frequent patterns among specified datasets
        # and use them as interested motifs
        # Properly handle different input formats for motifs
        if motifs is None:
            # If motifs is None, keep the existing logic to find patterns
            fp = self.find_fp_dist(ct=ct, dataset=dataset, max_dist=max_dist, min_size=min_size,
                                   min_support=min_support, max_ns=max_ns)
            motifs = fp['itemsets'].tolist()
        else:
            if isinstance(motifs, str):
                motifs = [[motifs]]
            elif isinstance(motifs, list) and all(isinstance(m, str) for m in motifs):
                motifs = [motifs]
            # At this point, motifs should be list[list[str]]

            # Filter out undefined cell types from each motif
            all_labels = pd.concat([s.labels for s in self.spatial_queries])
            labels_unique_all = set(all_labels.unique())

            filtered_motifs = []
            for motif in motifs:
                # Check which cell types in this motif are valid
                motif_exc = [m for m in motif if m not in labels_unique_all]
                if len(motif_exc) > 0:
                    print(f"Not found {motif_exc} in {dataset}! Ignoring them.")

                # Filter the current motif to only include valid cell types
                valid_motif = [m for m in motif if m in labels_unique_all]

                # Only include this motif if it has at least one valid cell type
                if len(valid_motif) > 0:
                    filtered_motifs.append(valid_motif)

            # Check if we have any valid motifs left
            if len(filtered_motifs) == 0:
                raise ValueError(f"All cell types in motifs are missing in {self.label_key}.")

            motifs = filtered_motifs

        # Initialize dictionaries to store cell IDs if requested
        motif_cell_ids = {}
        center_cell_ids = {}

        for motif in motifs:
            n_labels = 0
            n_ct = 0
            n_motif_labels = 0
            n_motif_ct = 0

            motif = list(motif) if not isinstance(motif, list) else motif
            sort_motif = sorted(motif)

            if return_cellID:
                motif_str = str(sort_motif)
                motif_cell_ids[motif_str] = {}
                center_cell_ids[motif_str] = {}

            for s in self.spatial_queries:
                if s.dataset.split('_')[0] not in dataset:
                    continue

                cell_pos = s.spatial_pos
                labels = np.array(s.labels)
                labels_unique = np.unique(labels)

                contain_motif = [m in labels_unique for m in motif]
                if not np.all(contain_motif):
                    n_labels += labels.shape[0]
                    n_ct += np.sum(labels == ct)
                    continue
                else:
                    n_labels += labels.shape[0]
                    _, matching_cells_indices = query_pattern(motif, s.grid_cell_types, s.grid_indices)
                    if not matching_cells_indices:
                        # if matching_cells_indices is empty, it indicates no motif are grouped together within upper limit of radius (500)
                        continue 
                    matching_cells_indices = np.concatenate([t for t in matching_cells_indices.values()])
                    matching_cells_indices = np.unique(matching_cells_indices)
                    matching_cells_indices.sort()

                    # print(f"number of cells skipped: {len(matching_cells_indices)}")
                    # print(f"proportion of cells searched: {len(matching_cells_indices) / len(s.spatial_pos)}")
                    idxs_in_grids = s.kd_tree.query_ball_point(
                        s.spatial_pos[matching_cells_indices],
                        r=max_dist,
                        return_sorted=False,
                        workers=-1
                    )

                    # using numppy
                    label_encoder = LabelEncoder()
                    int_labels = label_encoder.fit_transform(labels)
                    int_motifs = label_encoder.transform(np.array(motif))

                    num_cells = len(s.spatial_pos)
                    num_types = len(label_encoder.classes_)
                    # filter center out of neighbors
                    idxs_filter = [np.array(ids)[np.array(ids) != i][:min(max_ns, len(ids))] for i, ids in
                                   zip(matching_cells_indices, idxs_in_grids)]

                    num_matching_cells = len(matching_cells_indices)
                    flat_neighbors = np.concatenate(idxs_filter)
                    row_indices = np.repeat(np.arange(num_matching_cells), [len(neigh) for neigh in idxs_filter])
                    neighbor_labels = int_labels[flat_neighbors]

                    neighbor_matrix = np.zeros((num_matching_cells, num_types), dtype=int)
                    np.add.at(neighbor_matrix, (row_indices, neighbor_labels), 1)

                    motif_mask = np.all(neighbor_matrix[:, int_motifs] > 0, axis=1)
                    n_motif_labels += np.sum(motif_mask)

                    if ct in np.unique(labels):
                        int_ct = label_encoder.transform(np.array(ct, dtype=object, ndmin=1))
                        mask = int_labels[matching_cells_indices] == int_ct
                        center_mask = mask & motif_mask
                        n_motif_ct += np.sum(center_mask)
                        n_ct += np.sum(s.labels == ct)

                        if return_cellID:
                            # Get IDs of center cells with motif in neighborhood
                            center_indices = matching_cells_indices[center_mask]

                            if len(center_indices) > 0:
                                idxs_center = np.array(idxs_filter, dtype=object)[mask & motif_mask]
                                all_neighbors_center = np.concatenate(idxs_center)
                                motif_mask_all = np.isin(np.array(s.labels), motif)
                                valid_neighbors_center = all_neighbors_center[motif_mask_all[all_neighbors_center]]
                                id_motif_celltype = set(valid_neighbors_center)

                                motif_cell_ids[motif_str][s.dataset] = list(id_motif_celltype)
                                center_cell_ids[motif_str][s.dataset] = list(center_indices)

            if ct in motif:
                n_ct = round(n_ct / motif.count(ct))
            hyge = hypergeom(M=n_labels, n=n_ct, N=n_motif_labels)
            motif_out = {'center': ct, 'motifs': sort_motif, 'n_center_motif': n_motif_ct,
                         'n_center': n_ct, 'n_motif': n_motif_labels, 'expectation': hyge.mean(), 'p-values': hyge.sf(n_motif_ct)}
            out.append(motif_out)

        out_pd = pd.DataFrame(out)

        if len(out_pd) == 1:
            out_pd['if_significant'] = True if out_pd['p-values'][0] < 0.05 else False
        else:
            p_values = out_pd['p-values'].tolist()
            if_rejected, corrected_p_values = mt.fdrcorrection(p_values,
                                                               alpha=0.05,
                                                               method='poscorr')
            out_pd['adj_pvals'] = corrected_p_values
            out_pd['if_significant'] = if_rejected
            out_pd = out_pd.sort_values(by='adj_pvals', ignore_index=True)

        if return_cellID:
            return out_pd, motif_cell_ids, center_cell_ids
        else:
            return out_pd

    def find_fp_knn_fov(self,
                        ct: str,
                        dataset_i: str,
                        k: int = 30,
                        min_support: float = 0.5,
                        max_dist: float = 500.0
                        ) -> pd.DataFrame:
        """
        Find frequent patterns within the KNNs of specific cell type of interest in single field of view.

        Parameter
        ---------
        ct:
            Cell type name.
        dataset_i:
            Datasets for searching for frequent patterns in dataset_i format.
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
        if dataset_i not in self.datasets:
            raise ValueError(f"Found no {dataset_i.split('_')[0]} in any datasets.")

        max_dist = min(max_dist, self.max_radius)

        sp_object = self.spatial_queries[self.datasets.index(dataset_i)]
        cell_pos = sp_object.spatial_pos
        labels = np.array(sp_object.labels)
        if ct not in np.unique(labels):
            return pd.DataFrame(columns=['support', 'itemsets'])

        ct_pos = cell_pos[labels == ct]

        # Identify frequent patterns of cell types, including those subsets of patterns
        # whose support value exceeds min_support. Focus solely on the multiplicity
        # of cell types, rather than their frequency.
        fp, _, _ = build_fptree_knn(
            kd_tree=sp_object.kd_tree,
            labels=labels,
            max_radius=sp_object.max_radius,
            cell_pos=ct_pos,
            spatial_pos=cell_pos,
            k=k,
            min_support=min_support,
            if_max=False,
            max_dist=max_dist,
        )
        return fp

    def find_fp_dist_fov(self,
                         ct: str,
                         dataset_i: str,
                         max_dist: float = 100,
                         min_size: int = 0,
                         min_support: float = 0.5,
                         max_ns: int = 100
                         ):
        """
        Find frequent patterns within the radius-based neighborhood of specific cell type of interest
        in single field of view.

        Parameter
        ---------
        ct:
            Cell type name.
        dataset_i:
            Datasets for searching for frequent patterns in dataset_i format.
        max_dist:
            Maximum distance for considering a cell as a neighbor.
        min_size:
            Minimum neighborhood size for each point to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        max_ns:
            Maximum number of neighborhood size for each point.

        Return
        ------
            Frequent patterns in the neighborhood of certain cell type.
        """
        if dataset_i not in self.datasets:
            raise ValueError(f"Found no {dataset_i.split('_')[0]} in any datasets.")

        max_dist = min(max_dist, self.max_radius)

        sp_object = self.spatial_queries[self.datasets.index(dataset_i)]
        cell_pos = sp_object.spatial_pos
        labels = sp_object.labels
        if ct not in labels.unique():
            return pd.DataFrame(columns=['support, itemsets'])

        cinds = [id for id, l in enumerate(labels) if l == ct]
        ct_pos = cell_pos[cinds]

        fp, _, _ = build_fptree_dist(
            kd_tree=sp_object.kd_tree,
            labels=labels,
            cell_pos=ct_pos,
            spatial_pos=cell_pos,
            max_dist=max_dist,
            min_support=min_support,
            if_max=False,
            min_size=min_size,
            cinds=cinds,
            max_ns=max_ns,
        )
        return fp

    def differential_analysis_knn(self,
                                  ct: str,
                                  datasets: List[str],
                                  k: int = 30,
                                  min_support: float = 0.5,
                                  max_dist: float = 500,
                                  ):
        """
        Explore the differences in cell types and frequent patterns of cell types in spatial KNN neighborhood of cell
        type of interest. Perform differential analysis of frequent patterns in specified datasets.

        Parameter
        ---------
        ct:
            Cell type of interest as center point.
        datasets:
            Dataset names used to perform differential analysis
        k:
            Number of nearest neighbors.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        max_dist:
            Maximum distance for considering a cell as a neighbor.

        Return
        ------
            Dataframes with significant enriched patterns in differential analysis
        """
        if len(datasets) != 2:
            raise ValueError("Require 2 datasets for differential analysis.")

        max_dist = min(max_dist, self.max_radius)

        # Check if the two datasets are valid
        valid_ds_names = [s.dataset.split('_')[0] for s in self.spatial_queries]
        for ds in datasets:
            if ds not in valid_ds_names:
                raise ValueError(f"Invalid input dataset name: {ds}.\n"
                                 f"Valid dataset names are: {set(valid_ds_names)}")

        flag = 0
        # Identify frequent patterns in each dataset
        for d in datasets:
            fp_d = {}
            dataset_i = [ds for ds in self.datasets if ds.split('_')[0] == d]
            for d_i in dataset_i:
                fp_fov = self.find_fp_knn_fov(ct=ct,
                                              dataset_i=d_i,
                                              k=k,
                                              min_support=min_support,
                                              max_dist=max_dist)
                if len(fp_fov) > 0:
                    fp_d[d_i] = fp_fov

            if len(fp_d) == 1:
                common_patterns = list(fp_d.values())[0]
                common_patterns = common_patterns.rename(columns={'support': f"support_{list(fp_d.keys())[0]}"})
            else:
                # in comm_fps, duplicates items are not allowed by using set object
                comm_fps = set.intersection(
                    *[set(df['itemsets'].apply(lambda x: tuple(sorted(x)))) for df in
                      fp_d.values()])  # the items' order in patterns will not affect the returned intersection
                common_patterns = pd.DataFrame({'itemsets': [list(items) for items in comm_fps]})
                for data_name, df in fp_d.items():
                    support_dict = {itemset: support for itemset, support in
                                    df[['itemsets', 'support']].apply(
                                        lambda row: (tuple(sorted(row['itemsets'])), row['support']), axis=1)}
                    # support_dict = {tuple(itemset): support for itemset, support in df[['itemsets', 'support']].apply(
                    #     lambda row: (tuple(row['itemsets']), row['support']), axis=1)}
                    common_patterns[f"support_{data_name}"] = common_patterns['itemsets'].apply(
                        lambda x: support_dict.get(tuple(x), None))
            common_patterns['itemsets'] = common_patterns['itemsets'].apply(tuple)
            if flag == 0:
                fp_datasets = common_patterns
                flag = 1
            else:
                fp_datasets = fp_datasets.merge(common_patterns, how='outer', on='itemsets', ).fillna(0)

        match_ind_datasets = [
            [col for ind, col in enumerate(fp_datasets.columns) if col.startswith(f"support_{dataset}")] for dataset in
            datasets]
        p_values = []
        dataset_higher_ranks = []
        for index, row in fp_datasets.iterrows():
            group1 = pd.to_numeric(row[match_ind_datasets[0]].values)
            group2 = pd.to_numeric(row[match_ind_datasets[1]].values)

            # Perform the Mann-Whitney U test
            stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided', method='auto')
            p_values.append(p)

            # Label the dataset with higher frequency of patterns based on rank median
            support_rank = pd.concat([pd.DataFrame(group1), pd.DataFrame(group2)]).rank()  # ascending
            median_rank1 = support_rank[:len(group1)].median()[0]
            median_rank2 = support_rank[len(group1):].median()[0]
            if median_rank1 > median_rank2:
                dataset_higher_ranks.append(datasets[0])
            else:
                dataset_higher_ranks.append(datasets[1])

        fp_datasets['dataset_higher_frequency'] = dataset_higher_ranks
        # Apply Benjamini-Hochberg correction for multiple testing problems
        if_rejected, corrected_p_values = mt.fdrcorrection(p_values,
                                                           alpha=0.05,
                                                           method='poscorr')

        # Add the corrected p-values back to the DataFrame (optional)
        fp_datasets['adj_pvals'] = corrected_p_values
        fp_datasets['if_significant'] = if_rejected

        # Return the significant patterns in each dataset
        fp_dataset0 = fp_datasets[
            (fp_datasets['dataset_higher_frequency'] == datasets[0]) & (fp_datasets['if_significant'])
            ][['itemsets', 'adj_pvals']]
        fp_dataset1 = fp_datasets[
            (fp_datasets['dataset_higher_frequency'] == datasets[1]) & (fp_datasets['if_significant'])
            ][['itemsets', 'adj_pvals']]
        fp_dataset0 = fp_dataset0.reset_index(drop=True)
        fp_dataset1 = fp_dataset1.reset_index(drop=True)
        fp_dataset0 = fp_dataset0.sort_values(by='adj_pvals', ascending=True)
        fp_dataset1 = fp_dataset1.sort_values(by='adj_pvals', ascending=True)
        return {datasets[0]: fp_dataset0, datasets[1]: fp_dataset1}

    def differential_analysis_dist(self,
                                   ct: str,
                                   datasets: List[str],
                                   max_dist: float = 100,
                                   min_support: float = 0.5,
                                   min_size: int = 0,
                                   max_ns: int = 100,
                                   ):
        """
        Explore the differences in cell types and frequent patterns of cell types in spatial radius-based neighborhood
        of cell type of interest. Perform differential analysis of frequent patterns in specified datasets.

        Parameter
        ---------
        ct:
            Cell type of interest as center point.
        datasets:
            Dataset names used to perform differential analysis
        max_dist:
            Maximum distance for considering a cell as a neighbor.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        min_size:
            Minimum neighborhood size for each point to consider.
        max_ns:
            Upper limit of neighbors for each point.

        Return
        ------
            Dataframes with significant enriched patterns in differential analysis
        """
        if len(datasets) != 2:
            raise ValueError("Require 2 datasets for differential analysis.")

        # Check if the two datasets are valid
        valid_ds_names = [s.dataset.split('_')[0] for s in self.spatial_queries]
        for ds in datasets:
            if ds not in valid_ds_names:
                raise ValueError(f"Invalid input dataset name: {ds}.\n"
                                 f"Valid dataset names are: {set(valid_ds_names)}")

        max_dist = min(max_dist, self.max_radius)
        
        flag = 0
        # Identify frequent patterns in each dataset
        for d in datasets:
            fp_d = {}
            dataset_i = [ds for ds in self.datasets if ds.split('_')[0] == d]
            for d_i in dataset_i:
                fp_fov = self.find_fp_dist_fov(ct=ct,
                                               dataset_i=d_i,
                                               max_dist=max_dist,
                                               min_size=min_size,
                                               min_support=min_support,
                                               max_ns=max_ns)
                if len(fp_fov) > 0:
                    fp_d[d_i] = fp_fov

            if len(fp_d) == 1:
                common_patterns = list(fp_d.values())[0]
                common_patterns = common_patterns.rename(columns={'support': f"support_{list(fp_d.keys())[0]}"})
            else:
                comm_fps = set.intersection(*[set(df['itemsets'].apply(lambda x: tuple(sorted(x)))) for df in
                                              fp_d.values()])  # the items' order in patterns will not affect the returned intersection
                common_patterns = pd.DataFrame({'itemsets': [list(items) for items in comm_fps]})
                for data_name, df in fp_d.items():
                    support_dict = {itemset: support for itemset, support in df[['itemsets', 'support']].apply(
                        lambda row: (tuple(sorted(row['itemsets'])), row['support']), axis=1)}
                    common_patterns[f"support_{data_name}"] = common_patterns['itemsets'].apply(
                        lambda x: support_dict.get(tuple(x), None))
            common_patterns['itemsets'] = common_patterns['itemsets'].apply(tuple)
            if flag == 0:
                fp_datasets = common_patterns
                flag = 1
            else:
                fp_datasets = fp_datasets.merge(common_patterns, how='outer', on='itemsets', ).fillna(0)

        match_ind_datasets = [
            [col for ind, col in enumerate(fp_datasets.columns) if col.startswith(f"support_{dataset}")] for dataset in
            datasets]
        p_values = []
        dataset_higher_ranks = []
        for index, row in fp_datasets.iterrows():
            group1 = pd.to_numeric(row[match_ind_datasets[0]].values)
            group2 = pd.to_numeric(row[match_ind_datasets[1]].values)

            # Perform the Mann-Whitney U test
            stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided', method='auto')
            p_values.append(p)

            # Label the dataset with higher frequency of patterns based on rank sum
            support_rank = pd.concat([pd.DataFrame(group1), pd.DataFrame(group2)]).rank()  # ascending
            median_rank1 = support_rank[:len(group1)].median()[0]
            median_rank2 = support_rank[len(group1):].median()[0]
            if median_rank1 > median_rank2:
                dataset_higher_ranks.append(datasets[0])
            else:
                dataset_higher_ranks.append(datasets[1])

        fp_datasets['dataset_higher_frequency'] = dataset_higher_ranks
        # Apply Benjamini-Hochberg correction for multiple testing problems
        if_rejected, corrected_p_values = mt.fdrcorrection(p_values,
                                                           alpha=0.05,
                                                           method='poscorr')

        # Add the corrected p-values back to the DataFrame (optional)
        fp_datasets['adj_pvals'] = corrected_p_values
        fp_datasets['if_significant'] = if_rejected

        # Return the significant patterns in each dataset
        fp_dataset0 = fp_datasets[
            (fp_datasets['dataset_higher_frequency'] == datasets[0]) & (fp_datasets['if_significant'])
            ][['itemsets', 'adj_pvals']]
        fp_dataset1 = fp_datasets[
            (fp_datasets['dataset_higher_frequency'] == datasets[1]) & (fp_datasets['if_significant'])
            ][['itemsets', 'adj_pvals']]

        fp_dataset0 = fp_dataset0.sort_values(by='adj_pvals', ascending=True)
        fp_dataset1 = fp_dataset1.sort_values(by='adj_pvals', ascending=True)
        fp_dataset0 = fp_dataset0.reset_index(drop=True)
        fp_dataset1 = fp_dataset1.reset_index(drop=True)
        return {datasets[0]: fp_dataset0, datasets[1]: fp_dataset1}

    def de_genes(self,
                 ind_group1: Dict[str, List[int]],
                 ind_group2: Dict[str, List[int]],
                 genes: Optional[Union[str, List[str]]] = None,
                 min_fraction: float = 0.05,
                 method: Literal['fisher', 't-test', 'wilcoxon'] = 'fisher',
                 ) -> pd.DataFrame:
        """
        Perform differential expression analysis on the given indices.
        The ind_group1 and ind_group2 should be a defaultdict with keys as modified dataset names and values as
        lists of indices in corresponding group.
        It provides a flexible way to perform DE analysis on different datasets, e.g., across different FOVs of the
        same condition, or across FOVs from different conditions.

        Parameters
        ----------
        ind_group1: defaultdict[str, List[int]]
            A defaultdict with keys as modified dataset names and values as lists of indices in corresponding group.
        ind_group2: defaultdict[str, List[int]]
            A defaultdict with keys as modified dataset names and values as lists of indices in corresponding group.
        genes: Optional[Union[str, List[str]]]
            Genes to be searched in the gene index.
        min_fraction: float, default=0.05
            The minimum fraction of cells that express a gene for it to be considered differentially expressed.
        method: Literal['fisher', 't-test', 'wilcoxon'], default='fisher'
            The method to use for DE analysis. If build_gene_index=True, only Fisher's exact test is supported.

        Returns
        -------
        pd.DataFrame
            DataFrame containing differential expression results.
        """
        if self.build_gene_index:
            # Use scfind index-based method with Fisher's exact test
            if method != 'fisher':
                print(f"Warning: When build_gene_index=True, only Fisher's exact test is supported. Ignoring method='{method}'.")
            return self._de_genes_scfind(ind_group1, ind_group2, genes, min_fraction)
        else:
            # Use adata.X directly with specified method
            return self._de_genes_adata(ind_group1, ind_group2, genes, min_fraction, method)

    def _de_genes_scfind(self,
                         ind_group1: Dict[str, List[int]],
                         ind_group2: Dict[str, List[int]],
                         genes: Optional[Union[str, List[str]]] = None,
                         min_fraction: float = 0.05
                         ) -> pd.DataFrame:
        """
        Perform differential expression analysis using scfind index with Fisher's exact test.
        This method is used when build_gene_index=True.
        """

        # For each gene, calculate the number of cells in the provided indices expressing the gene in each group
        if genes is None:
            genes = set.intersection(*[set(s.genes) for s in self.spatial_queries])
            genes = list(genes)
            print(f"Testing {len(genes)} genes with Fisher's exact test ...\n")

        n_1 = np.sum([len(ids) for ids in ind_group1.values()])
        n_2 = np.sum([len(ids) for ids in ind_group2.values()])

        valid_ds1 = [ds for ds in ind_group1.keys() if ds in self.datasets]
        valid_ds2 = [ds for ds in ind_group2.keys() if ds in self.datasets]

        # Check if there are valid datasets
        if not valid_ds1:
            raise ValueError("No valid datasets found in ind_group1.")
        if not valid_ds2:
            raise ValueError("No valid datasets found in ind_group2.")

        group1_results = []
        start = time()
        for ds, ids in ind_group1.items():
            print(f"Processing {ds} in group1...")
            if ds not in valid_ds1:
                print(f'Warning: {ds} is not a valid dataset name. Ignoring it.')
                continue

            ds_i = self.datasets.index(ds)
            sp = self.spatial_queries[ds_i]

            # Get counts of cells expressing each gene in this dataset
            start1 = time()
            genes_sp = sp.index._case_correct(genes, if_print=False)
            if not genes_sp:
                continue
                
            ds_counts = sp.index.index.cell_counts_in_indices_genes(ids, genes_sp)
            end1 = time()
            print(f'Cell search in dataset {ds}: {end1 - start1:.2f} seconds')

            genes_list = [item['gene'] for item in ds_counts]
            counts_list = [item['expressed_cells'] for item in ds_counts]

            # Create a DataFrame
            if genes_list:
                temp_df = pd.DataFrame({'gene': genes_list, 'count': counts_list})
                group1_results.append(temp_df)

        end = time()
        print(f'Time for cell search in group1: {end - start}')
        # Count cells expressing each gene in group 2
        group2_results = []
        start = time()
        for ds, ids in ind_group2.items():
            print(f"Processing {ds} in group2...")
            if ds not in valid_ds2:
                print(f'Warning: {ds} is not a valid dataset name. Ignoring it.')
                continue

            ds_i = self.datasets.index(ds)
            sp = self.spatial_queries[ds_i]

            # Get counts of cells expressing each gene in this dataset
            genes_sp = sp.index._case_correct(genes, if_print=False)
            if not genes_sp:
                continue

            ds_counts = sp.index.index.cell_counts_in_indices_genes(ids, genes_sp)

            genes_list = [item['gene'] for item in ds_counts]
            counts_list = [item['expressed_cells'] for item in ds_counts]

            # Create a DataFrame in one operation
            if genes_list:
                temp_df = pd.DataFrame({'gene': genes_list, 'count': counts_list})
                group2_results.append(temp_df)

        end = time()
        print(f'Time for cell search in group2: {end - start:.2f} seconds')

        # Prepare data for statistical testing
        # Combine all results
        # start = time()
        if group1_results:
            group1_df = pd.concat(group1_results, ignore_index=True)
            group1_agg = group1_df.groupby('gene')['count'].sum().reset_index()
            group1_agg = group1_agg.rename(columns={'count': 'count_1'})
        else:
            group1_agg = pd.DataFrame(columns=['gene', 'count_1'])

        if group2_results:
            group2_df = pd.concat(group2_results, ignore_index=True)
            group2_agg = group2_df.groupby('gene')['count'].sum().reset_index()
            group2_agg = group2_agg.rename(columns={'count': 'count_2'})
        else:
            group2_agg = pd.DataFrame(columns=['gene', 'count_2'])

        # Merge the two groups
        merged_df = pd.merge(group1_agg, group2_agg, on='gene', how='outer').fillna(0)

        # Calculate proportions
        merged_df['proportion_1'] = merged_df['count_1'] / n_1
        merged_df['proportion_2'] = merged_df['count_2'] / n_2

        # Filter by minimum fraction
        filtered_df = merged_df[(merged_df['proportion_1'] >= min_fraction) |
                                (merged_df['proportion_2'] >= min_fraction)].copy()

        if filtered_df.empty:
            print("No genes meet the minimum fraction threshold.")
            return pd.DataFrame(
                columns=["gene", "proportion_1", "proportion_2", "abs",
                         "difference", "p_value", "adj_p_value", "de_in"]
            )

        # Calculate differences
        filtered_df.loc[:, 'difference'] = filtered_df['proportion_1'] - filtered_df['proportion_2']
        filtered_df.loc[:, 'abs'] = filtered_df['difference'].abs()

        # For Fisher's exact test, prepare arrays for vectorized operations
        count_1_array = filtered_df['count_1'].values.astype(int)
        count_2_array = filtered_df['count_2'].values.astype(int)
        not_count_1_array = n_1 - count_1_array
        not_count_2_array = n_2 - count_2_array
        # end = time()
        # print(f'Time for prepare data for preparation of statistical testing: {end - start}')

        # Use numpy to create all contingency tables at once
        # This creates a 3D array of shape (n_rows, 2, 2)
        # start = time()
        contingency_tables = np.array([
            [[count_1_array[i], not_count_1_array[i]],
             [count_2_array[i], not_count_2_array[i]]]
            for i in range(len(count_1_array))
        ])

        # Apply Fisher's exact test - this still needs a loop but is more efficient
        # Use parallelization if available (requires joblib)
        def apply_fisher(table):
            _, p_value = stats.fisher_exact(table, alternative='two-sided')
            return p_value

        # Run tests in parallel
        # n_jobs=-1 uses all available cores
        # p_values_parallel = Parallel(n_jobs=-1)(
        #     delayed(apply_fisher)(table) for table in contingency_tables
        # )
        # Also compute p_values sequentially (without Parallel)
        p_values = [apply_fisher(table) for table in contingency_tables]

        # Add p-values to DataFrame
        filtered_df.loc[:, 'p_value'] = p_values

        # Sort by p-value
        filtered_df = filtered_df.sort_values('p_value')

        # Multiple testing correction
        if len(filtered_df) > 1:
            adjusted_pvals = multipletests(filtered_df['p_value'], method='fdr_bh')[1]
            filtered_df['adj_p_value'] = adjusted_pvals
        else:
            filtered_df['adj_p_value'] = filtered_df['p_value']

        filtered_df = filtered_df[filtered_df['adj_p_value']<0.05].reset_index(drop=True)

        # Add information about which group shows higher expression
        filtered_df['de_in'] = np.where(
            (filtered_df['proportion_1'] > filtered_df['proportion_2']),
            'group1',
            np.where(
                (filtered_df['proportion_2'] > filtered_df['proportion_1']),
                'group2',
                None
            )
        )
        # end = time()
        # print(f'Time for statistical testing: {end - start}')

        # Return the final results
        return filtered_df[["gene", "proportion_1", "proportion_2", "abs",
                            "difference", "p_value", "adj_p_value", "de_in"]]

    def _de_genes_adata(self,
                        ind_group1: Dict[str, List[int]],
                        ind_group2: Dict[str, List[int]],
                        genes: Optional[Union[str, List[str]]] = None,
                        min_fraction: float = 0.05,
                        method: Literal['fisher', 't-test', 'wilcoxon'] = 'fisher',
                        ) -> pd.DataFrame:
        """
        Perform differential expression analysis using adata.X directly.
        This method is used when build_gene_index=False.
        
        For each dataset, collect the cells and concatenate them, then perform DE analysis.
        """
        # Validate datasets
        valid_ds1 = [ds for ds in ind_group1.keys() if ds in self.datasets]
        valid_ds2 = [ds for ds in ind_group2.keys() if ds in self.datasets]
        
        if not valid_ds1:
            raise ValueError("No valid datasets found in ind_group1.")
        if not valid_ds2:
            raise ValueError("No valid datasets found in ind_group2.")
        
        # Collect all cells from group 1
        all_adatas_g1 = []
        
        for ds in valid_ds1:
            if ds not in ind_group1 or len(ind_group1[ds]) == 0:
                continue
            
            ds_i = self.datasets.index(ds)
            sp = self.spatial_queries[ds_i]
            
            # Get the adata for this dataset
            if sp.adata is None:
                raise ValueError(f"Error: {ds} does not have adata.X. Please use use fisher's exact using indexed data.")
            
            # Extract cells for group 1
            idx_g1 = ind_group1[ds]
            adata_subset = sp.adata[idx_g1].copy()
            all_adatas_g1.append(adata_subset)
        
        # Collect all cells from group 2
        all_adatas_g2 = []
        
        for ds in valid_ds2:
            if ds not in ind_group2 or len(ind_group2[ds]) == 0:
                continue
            
            ds_i = self.datasets.index(ds)
            sp = self.spatial_queries[ds_i]
            
            # Get the adata for this dataset
            if sp.adata is None:
                raise ValueError(f"Error: {ds} does not have adata.X. Please use use fisher's exact using indexed data.")
                continue
            
            # Extract cells for group 2
            idx_g2 = ind_group2[ds]
            adata_subset = sp.adata[idx_g2].copy()
            all_adatas_g2.append(adata_subset)
        
        if not all_adatas_g1:
            raise ValueError("No valid adata found in group 1.")
        if not all_adatas_g2:
            raise ValueError("No valid adata found in group 2.")

        
        adata_g1 = ad.concat(all_adatas_g1, join='inner')
        adata_g2 = ad.concat(all_adatas_g2, join='inner')
        
        # Combine both groups
        adata_combined = ad.concat([adata_g1, adata_g2], join='inner')

        # Get the overlapping genes of each data
        genes_list = adata_combined.var_names.tolist()

        # Create indices for combined adata
        ind_combined_g1 = list(range(len(adata_g1)))
        ind_combined_g2 = list(range(len(adata_g1), len(adata_combined)))
        
        # Perform DE analysis using spatial_utils
        if method == 'fisher':
            results_df = de_genes_fisher(
                adata_combined, genes_list, ind_combined_g1, ind_combined_g2, genes, min_fraction
            )
        elif method == 't-test' or method == 'wilcoxon':
            results_df = de_genes_scanpy(
                adata_combined, genes_list, ind_combined_g1, ind_combined_g2, genes, min_fraction, method=method
            )
        else:
            raise ValueError(f"Invalid method: {method}. Choose from 'fisher', 't-test', or 'wilcoxon'.")
        
        return results_df

    def cell_type_distribution(self,
                               dataset: Union[str, List[str]] = None,
                               data_type: str = 'number',
                               ):
        """
        Visualize the distribution of cell types across datasets using a stacked bar plot.

        Parameter
        ---------
        dataset:
            Datasets for searching.
        data_type:
            Plot bar plot by number of cells or by the proportions of datasets in each cell type.
            Default is 'number' otherwise 'proportion' is used.
        Returns
        -------
        Stacked bar plot
        """
        if data_type not in ['number', 'proportion']:
            raise ValueError("Invalild data_type. It should be one of 'number' or 'proportion'.")

        if dataset is None:
            dataset = [s.dataset.split('_')[0] for s in self.spatial_queries]
        if isinstance(dataset, str):
            dataset = [dataset]

        valid_ds_names = [s.dataset.split('_')[0] for s in self.spatial_queries]
        for ds in dataset:
            if ds not in valid_ds_names:
                raise ValueError(f"Invalid input dataset name: {ds}.\n "
                                 f"Valid dataset names are: {set(valid_ds_names)}")

        summary = defaultdict(lambda: defaultdict(int))

        valid_queries = [s for s in self.spatial_queries if s.dataset.split('_')[0] in dataset]
        cell_types = set([ct for s in valid_queries for ct in s.labels.unique()])
        for s in valid_queries:
            for cell_type in cell_types:
                summary[s.dataset][cell_type] += np.sum(s.labels == cell_type)

        df = pd.DataFrame([(dataset, cell_type, count)
                           for dataset, cell_types in summary.items()
                           for cell_type, count in cell_types.items()],
                          columns=['Dataset', 'Cell Type', 'Count'])

        df['dataset'] = df['Dataset'].str.split('_').str[0]

        summary = df.groupby(['dataset', 'Cell Type'])['Count'].sum().reset_index()
        plot_data = summary.pivot(index='Cell Type', columns='dataset', values='Count').fillna(0)

        # Sort the cell types by total count (descending)
        plot_data = plot_data.sort_values(by=plot_data.columns.tolist(), ascending=False, )

        if data_type != 'number':
            plot_data = plot_data.div(plot_data.sum(axis=1), axis=0)

        # Create the stacked bar plot
        ax = plot_data.plot(kind='bar', stacked=True,
                            figsize=(plot_data.shape[0], plot_data.shape[0] * 0.6),
                            edgecolor='black')

        # Customize the plot
        plt.title(f"Distribution of Cell Types Across Datasets", fontsize=16)
        plt.xlabel('Cell Types', fontsize=12)
        if data_type == 'number':
            plt.ylabel('Number of Cells', fontsize=12)
        else:
            plt.ylabel('Proportion of Cells', fontsize=12)

        plt.xticks(rotation=90, ha='right', fontsize=10)

        plt.legend(title='Datasets', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

    def cell_type_distribution_fov(self,
                                   dataset: str,
                                   data_type: str = 'number',
                                   ):
        """
        Visualize the distribution of cell types across FOVs in the dataset using a stacked bar plot.
        Parameter
        ---------
        dataset:
            Dataset of searching.
        data_type:
            Plot bar plot by number of cells or by the proportions of cell types in each FOV.
            Default is 'number' otherwise 'proportion' is used.
        Returns
        -------
        Stacked bar plot
        """
        if data_type not in ['number', 'proportion']:
            raise ValueError("Invalild data_type. It should be one of 'number' or 'proportion'.")

        valid_ds_names = [s.dataset.split('_')[0] for s in self.spatial_queries]
        if dataset not in valid_ds_names:
            raise ValueError(f"Invalid input dataset name: {dataset}. \n"
                             f"Valid dataset names are: {set(valid_ds_names)}")
        valid_queries = [s for s in self.spatial_queries if s.dataset.split('_')[0] == dataset]
        cell_types = set([ct for s in valid_queries for ct in s.labels.unique()])

        summary = defaultdict(lambda: defaultdict(int))
        for s in valid_queries:
            for cell_type in cell_types:
                summary[s.dataset][cell_type] = np.sum(s.labels == cell_type)

        df = pd.DataFrame([(dataset, cell_type, count)
                           for dataset, cell_types in summary.items()
                           for cell_type, count in cell_types.items()],
                          columns=['Dataset', 'Cell Type', 'Count'])

        df['FOV'] = df['Dataset'].str.split('_').str[1]

        summary = df.groupby(['FOV', 'Cell Type'])['Count'].sum().reset_index()
        plot_data = summary.pivot(columns='Cell Type', index='FOV', values='Count').fillna(0)

        # Sort the cell types by total count (descending)
        row_sums = plot_data.sum(axis=1)
        plot_data_sorted = plot_data.loc[row_sums.sort_values(ascending=False).index]

        if data_type != 'number':
            plot_data_sorted = plot_data_sorted.div(plot_data_sorted.sum(axis=1), axis=0)

            # Create the stacked bar plot
        ax = plot_data_sorted.plot(kind='bar', stacked=True,
                                   figsize=(plot_data.shape[0] * 0.6, plot_data.shape[0] * 0.3),
                                   edgecolor='black')

        # Customize the plot
        plt.title(f"Distribution of FOVs in {dataset} dataset", fontsize=20)
        plt.xlabel('FOV', fontsize=12)
        if data_type == 'number':
            plt.ylabel('Number of Cells', fontsize=12)
        else:
            plt.ylabel('Proportion of Cells', fontsize=12)

        plt.xticks(rotation=90, ha='right', fontsize=10)

        plt.legend(title='Cell Type', bbox_to_anchor=(1, 1.05), loc='center left', fontsize=12)

        plt.tight_layout()
        plt.show()

    def compute_gene_gene_correlation(self,
                                       ct: str,
                                       motif: Union[str, List[str]],
                                       dataset: Union[str, List[str]] = None,
                                       genes: Optional[Union[str, List[str]]] = None,
                                       max_dist: Optional[float] = None,
                                       k: Optional[int] = None,
                                       min_size: int = 0,
                                       min_nonzero: int = 10,
                                       ) -> Tuple[pd.DataFrame, Dict]:
        """
        Compute gene-gene co-varying patterns between motif and center cells across multiple FOVs.

        Similar to compute_gene_gene_correlation in single FOV, but:
        - Aggregates center-neighbor pairs across all FOVs in specified dataset
        - Uses FOV-specific cell type means for centering (NOT global means)
        - Computes correlations by accumulating statistics across FOVs

        This function calculates cross correlation between gene expression in:
        1. Motif cells that are neighbors of center cell type (paired data across FOVs)
        2. Motif cells that are NOT neighbors of center cell type (all-to-all across FOVs)
        3. Neighboring cells of center cell type without nearby motif (paired data across FOVs)

        Parameter
        ---------
        ct:
            Cell type as the center cells.
        motif:
            Motif (names of cell types) to be analyzed.
        dataset:
            Datasets to include in analysis. If None, use all datasets.
        genes:
            List of genes to analyze. If None, uses intersection of genes across all FOVs.
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
        results_df : pd.DataFrame
            DataFrame with correlation results between neighbor and non-neighbor groups.
            Columns include:
                - gene_center, gene_motif: gene pairs
                - corr_neighbor: correlation in neighbor group
                - corr_non_neighbor: correlation in non-neighbor group
                - corr_center_no_motif: correlation for centers without motif
                - p_value_test1: p-value for test1 (neighbor vs non-neighbor)
                - p_value_test2: p-value for test2 (with motif vs without motif)
                - delta_corr_test1, delta_corr_test2: correlation differences
                - combined_score: combined significance score
                - adj_p_value_test1, adj_p_value_test2: FDR-corrected p-values

        fov_info : Dict
            Dictionary containing FOV-level information:
                - 'fov_statistics': detailed statistics from each FOV
                - 'total_pairs_neighbor': total number of neighbor pairs
                - 'total_pairs_non_neighbor': total number of non-neighbor pairs
                - 'total_pairs_no_motif': total number of no-motif pairs
                - 'n_fovs_analyzed': number of FOVs included
        """

        # Validate parameters
        if (max_dist is None and k is None) or (max_dist is not None and k is not None):
            raise ValueError("Please specify either max_dist or k, but not both.")

        # Convert motif to list
        motif = motif if isinstance(motif, list) else [motif]

        # Validate and prepare dataset list
        if dataset is None:
            dataset = [s.dataset.split('_')[0] for s in self.spatial_queries]
            print(f"No dataset specified. Using all datasets.")
        if isinstance(dataset, str):
            dataset = [dataset]

        valid_ds_names = [s.dataset.split('_')[0] for s in self.spatial_queries]
        for ds in dataset:
            if ds not in valid_ds_names:
                raise ValueError(f"Invalid input dataset name: {ds}.\n "
                               f"Valid dataset names are: {set(valid_ds_names)}")

        # Filter spatial_queries to include only selected datasets
        selected_queries = [s for s in self.spatial_queries if s.dataset.split('_')[0] in dataset]

        # Check if ct and motif exist in at least one FOV
        ct_exists = any(ct in s.labels.unique() for s in selected_queries)
        if not ct_exists:
            raise ValueError(f"Center type '{ct}' not found in any selected datasets!")

        motif_exists = all(any(m in s.labels.unique() for s in selected_queries) for m in motif)
        if not motif_exists:
            missing = [m for m in motif if not any(m in s.labels.unique() for s in selected_queries)]
            raise ValueError(f"Motif types {missing} not found in any selected datasets!")

        # Get intersection of genes across all FOVs
        genes_sets = [set(sq.genes) for sq in selected_queries]
        all_genes = list(set.intersection(*genes_sets))
        if genes is None:    
            print(f"No genes specified. Using all common genes across all selected FOVs ...")
            valid_genes = all_genes
        elif isinstance(genes, str):
            genes = [genes]
            valid_genes = [g for g in genes if g in all_genes]
        else:
            valid_genes = [g for g in genes if g in all_genes]
        
        if len(valid_genes) == 0:
            raise ValueError("No valid genes found across all FOVs.")

        genes = valid_genes
        n_genes = len(genes)
        print(f"Analyzing {n_genes} genes across {len(selected_queries)} FOVs")

        # ====================================================================================
        # Step 1: FOV-level computation - collect statistics from each FOV
        # ====================================================================================
        print("\n" + "="*80)
        print("Step 1: Computing and accumulating statistics across FOVs")
        print("="*80)

        # Initialize accumulators for aggregated statistics
        # Neighbor (Correlation 1)
        total_cov_sum_neighbor = np.zeros((n_genes, n_genes))
        total_center_ss_neighbor = np.zeros(n_genes)
        total_neighbor_ss_neighbor = np.zeros(n_genes)
        total_n_pairs_neighbor = 0
        total_n_eff_neighbor = 0
        n_fovs_neighbor = 0

        # Non-neighbor (Correlation 2)
        total_cov_sum_non = np.zeros((n_genes, n_genes))
        total_center_ss_non = np.zeros(n_genes)
        total_non_neighbor_ss = np.zeros(n_genes)
        total_n_pairs_non = 0
        total_n_eff_non = 0
        n_fovs_non = 0

        # No-motif (Correlation 3)
        total_cov_sum_no_motif = np.zeros((n_genes, n_genes))
        total_center_ss_no_motif = np.zeros(n_genes)
        total_neighbor_ss_no_motif = np.zeros(n_genes)
        total_n_pairs_no_motif = 0
        total_n_eff_no_motif = 0
        n_fovs_no_motif = 0

        for fov_idx, sq in enumerate(selected_queries):
            print(f"\n--- Processing FOV {fov_idx + 1}/{len(selected_queries)}: {sq.dataset} ---")

            # Check if ct and all motif types exist in this FOV
            if ct not in sq.labels.unique():
                print(f"  Skipping: center type '{ct}' not in this FOV")
                continue

            missing_motif = [m for m in motif if m not in sq.labels.unique()]
            if missing_motif:
                print(f"  Skipping: motif types {missing_motif} not in this FOV")
                continue

            # Get expression data for this FOV
            expr_genes = sq.adata[:, genes].X
            is_sparse = sparse.issparse(expr_genes)

            # Filter genes by non-zero expression in this FOV
            if is_sparse:
                nonzero_fov = np.array((expr_genes > 0).sum(axis=0)).flatten()
            else:
                nonzero_fov = (expr_genes > 0).sum(axis=0)

            valid_gene_mask = nonzero_fov >= min_nonzero

            if valid_gene_mask.sum() < 10:
                print(f"  Skipping: only {valid_gene_mask.sum()} genes pass min_nonzero filter")
                continue

            # Note: We keep all genes for consistency across FOVs, but track which are valid
            # We'll filter at the end based on aggregate statistics

            # Compute FOV-specific cell type means
            fov_cell_type_means = {}
            for cell_type in sq.labels.unique():
                ct_mask = sq.labels == cell_type
                ct_cells = np.where(ct_mask)[0]
                if len(ct_cells) > 0:
                    ct_expr = expr_genes[ct_cells, :]
                    if is_sparse:
                        fov_cell_type_means[cell_type] = np.array(ct_expr.mean(axis=0)).flatten()
                    else:
                        fov_cell_type_means[cell_type] = ct_expr.mean(axis=0)

            center_mean = fov_cell_type_means[ct]

            # ========================================================================
            # Correlation 1: Center with motif vs Neighboring motif (PAIRED)
            # ========================================================================
            try:
                neighbor_result = spatial_utils.get_motif_neighbor_cells(
                    sq_obj=sq, ct=ct, motif=motif, max_dist=max_dist, k=k, min_size=min_size
                )
                center_neighbor_pairs = neighbor_result['center_neighbor_pairs']

                if len(center_neighbor_pairs) < 10:
                    print(f"  Skipping Corr1: only {len(center_neighbor_pairs)} pairs found")
                else:
                    print(f"  Corr1: {len(center_neighbor_pairs)} center-neighbor pairs")

                    # Extract pair indices
                    pair_centers = center_neighbor_pairs[:, 0]
                    pair_neighbors = center_neighbor_pairs[:, 1]

                    # Get neighbor cell types
                    neighbor_types = sq.labels[pair_neighbors]

                    # Compute statistics using optimized sparse-aware function
                    cov_sum, center_ss, neighbor_ss, n_pairs, n_eff = spatial_utils.compute_covariance_statistics_paired(
                        expr_genes=expr_genes,
                        pair_centers=pair_centers,
                        pair_neighbors=pair_neighbors,
                        center_mean=center_mean,
                        cell_type_means=fov_cell_type_means,
                        neighbor_cell_types=neighbor_types,
                        is_sparse=is_sparse
                    )

                    # Accumulate statistics directly (no intermediate storage)
                    total_cov_sum_neighbor += cov_sum
                    total_center_ss_neighbor += center_ss
                    total_neighbor_ss_neighbor += neighbor_ss
                    total_n_pairs_neighbor += n_pairs
                    total_n_eff_neighbor += n_eff
                    n_fovs_neighbor += 1

                    # Check for overflow/invalid values
                    if np.isinf(total_cov_sum_neighbor).any() or np.isnan(total_cov_sum_neighbor).any():
                        raise ValueError(f"Overflow or NaN detected in neighbor covariance sum at FOV {fov_idx + 1}")
                    if np.isinf(total_center_ss_neighbor).any() or np.isnan(total_center_ss_neighbor).any():
                        raise ValueError(f"Overflow or NaN detected in neighbor center_ss at FOV {fov_idx + 1}")
                    if np.isinf(total_neighbor_ss_neighbor).any() or np.isnan(total_neighbor_ss_neighbor).any():
                        raise ValueError(f"Overflow or NaN detected in neighbor neighbor_ss at FOV {fov_idx + 1}")

                    # ========================================================================
                    # Correlation 2: Center with motif vs Distant motif (ALL-TO-ALL)
                    # ========================================================================
                    # Get non-neighbor motif cells
                    motif_mask = np.isin(sq.labels.values, motif)
                    all_motif_cells = np.where(motif_mask)[0]
                    neighbor_cells_in_fov = np.unique(pair_neighbors)
                    non_neighbor_cells = np.setdiff1d(all_motif_cells, neighbor_cells_in_fov)

                    # Remove center type from non-neighbor
                    ct_in_motif = ct in motif
                    if ct_in_motif:
                        non_neighbor_cells = non_neighbor_cells[sq.labels[non_neighbor_cells] != ct]

                    if len(non_neighbor_cells) >= 10:
                        # Use unique center cells
                        unique_centers = np.unique(pair_centers)

                        print(f"  Corr2: {len(unique_centers)} centers × {len(non_neighbor_cells)} non-neighbors")

                        # Get non-neighbor types
                        non_neighbor_types = sq.labels[non_neighbor_cells]

                        # Compute statistics using optimized sparse-aware function
                        cov_sum_non, center_ss_non, non_neighbor_ss, n_pairs_non, n_eff_non = spatial_utils.compute_covariance_statistics_all_to_all(
                            expr_genes=expr_genes,
                            center_cells=unique_centers,
                            neighbor_cells=non_neighbor_cells,
                            center_mean=center_mean,
                            cell_type_means=fov_cell_type_means,
                            neighbor_cell_types=non_neighbor_types,
                            is_sparse=is_sparse
                        )

                        # Accumulate statistics directly
                        total_cov_sum_non += cov_sum_non
                        total_center_ss_non += center_ss_non
                        total_non_neighbor_ss += non_neighbor_ss
                        total_n_pairs_non += n_pairs_non
                        total_n_eff_non += n_eff_non
                        n_fovs_non += 1

                        # Check for overflow/invalid values
                        if np.isinf(total_cov_sum_non).any() or np.isnan(total_cov_sum_non).any():
                            raise ValueError(f"Overflow or NaN detected in non-neighbor covariance sum at FOV {fov_idx + 1}")
                        if np.isinf(total_center_ss_non).any() or np.isnan(total_center_ss_non).any():
                            raise ValueError(f"Overflow or NaN detected in non-neighbor center_ss at FOV {fov_idx + 1}")
                        if np.isinf(total_non_neighbor_ss).any() or np.isnan(total_non_neighbor_ss).any():
                            raise ValueError(f"Overflow or NaN detected in non-neighbor non_neighbor_ss at FOV {fov_idx + 1}")
                    else:
                        print(f"  Skipping Corr2: only {len(non_neighbor_cells)} non-neighbor cells")

                    # ========================================================================
                    # Correlation 3: Center without motif vs Neighbors (PAIRED)
                    # ========================================================================
                    no_motif_result = spatial_utils.get_all_neighbor_cells(
                        sq_obj=sq,
                        ct=ct,
                        max_dist=max_dist,
                        k=k,
                        min_size=min_size,
                        exclude_centers=np.unique(pair_centers),
                        exclude_neighbors=neighbor_cells_in_fov,
                    )

                    center_no_motif_pairs = no_motif_result['center_neighbor_pairs']

                    if len(center_no_motif_pairs) >= 10:
                        print(f"  Corr3: {len(center_no_motif_pairs)} no-motif pairs")

                        pair_centers_no_motif = center_no_motif_pairs[:, 0]
                        pair_neighbors_no_motif = center_no_motif_pairs[:, 1]

                        neighbor_no_motif_types = sq.labels[pair_neighbors_no_motif]

                        # Compute statistics using optimized sparse-aware function
                        cov_sum_no_motif, center_ss_no_motif, neighbor_ss_no_motif, n_pairs_no_motif, n_eff_no_motif = spatial_utils.compute_covariance_statistics_paired(
                            expr_genes=expr_genes,
                            pair_centers=pair_centers_no_motif,
                            pair_neighbors=pair_neighbors_no_motif,
                            center_mean=center_mean,
                            cell_type_means=fov_cell_type_means,
                            neighbor_cell_types=neighbor_no_motif_types,
                            is_sparse=is_sparse
                        )

                        # Accumulate statistics directly
                        total_cov_sum_no_motif += cov_sum_no_motif
                        total_center_ss_no_motif += center_ss_no_motif
                        total_neighbor_ss_no_motif += neighbor_ss_no_motif
                        total_n_pairs_no_motif += n_pairs_no_motif
                        total_n_eff_no_motif += n_eff_no_motif
                        n_fovs_no_motif += 1

                        # Check for overflow/invalid values
                        if np.isinf(total_cov_sum_no_motif).any() or np.isnan(total_cov_sum_no_motif).any():
                            raise ValueError(f"Overflow or NaN detected in no-motif covariance sum at FOV {fov_idx + 1}")
                        if np.isinf(total_center_ss_no_motif).any() or np.isnan(total_center_ss_no_motif).any():
                            raise ValueError(f"Overflow or NaN detected in no-motif center_ss at FOV {fov_idx + 1}")
                        if np.isinf(total_neighbor_ss_no_motif).any() or np.isnan(total_neighbor_ss_no_motif).any():
                            raise ValueError(f"Overflow or NaN detected in no-motif neighbor_ss at FOV {fov_idx + 1}")
                    else:
                        print(f"  Skipping Corr3: only {len(center_no_motif_pairs)} no-motif pairs")

            except Exception as e:
                print(f"  Error processing FOV: {e}")
                continue

        # ====================================================================================
        # Step 2: Validate and summarize accumulated statistics
        # ====================================================================================
        print("\n" + "="*80)
        print("Step 2: Summary of accumulated statistics")
        print("="*80)

        if n_fovs_neighbor == 0:
            raise ValueError("No valid neighbor pairs found across any FOV!")

        print(f"Correlation 1 (neighbor): {total_n_pairs_neighbor} total pairs, n_eff={total_n_eff_neighbor} from {n_fovs_neighbor} FOVs")

        if n_fovs_non > 0:
            print(f"Correlation 2 (non-neighbor): {total_n_pairs_non} total pairs, n_eff={total_n_eff_non} from {n_fovs_non} FOVs")
        else:
            print("Warning: No non-neighbor pairs found across any FOV!")
            total_cov_sum_non = None

        if n_fovs_no_motif > 0:
            print(f"Correlation 3 (no-motif): {total_n_pairs_no_motif} total pairs, n_eff={total_n_eff_no_motif} from {n_fovs_no_motif} FOVs")
        else:
            print("Warning: No no-motif pairs found across any FOV!")
            total_cov_sum_no_motif = None

        # ====================================================================================
        # Step 3: Compute correlation matrices
        # ====================================================================================
        print("\n" + "="*80)
        print("Step 3: Computing correlation matrices")
        print("="*80)

        # Correlation 1
        denominator_neighbor = np.sqrt(
            total_center_ss_neighbor[:, None] * total_neighbor_ss_neighbor[None, :]
        )
        corr_matrix_neighbor = total_cov_sum_neighbor / (denominator_neighbor + 1e-10)
        n_eff_neighbor = total_n_eff_neighbor  # Use accumulated n_eff

        print(f"Corr1 matrix shape: {corr_matrix_neighbor.shape}, effective n={n_eff_neighbor}")

        # Correlation 2
        if total_cov_sum_non is not None:
            denominator_non = np.sqrt(
                total_center_ss_non[:, None] * total_non_neighbor_ss[None, :]
            )
            corr_matrix_non_neighbor = total_cov_sum_non / (denominator_non + 1e-10)
            n_eff_non_neighbor = total_n_eff_non  # Use accumulated n_eff
            print(f"Corr2 matrix shape: {corr_matrix_non_neighbor.shape}, effective n={n_eff_non_neighbor}")
        else:
            corr_matrix_non_neighbor = np.zeros((n_genes, n_genes))
            n_eff_non_neighbor = 0

        # Correlation 3
        if total_cov_sum_no_motif is not None:
            denominator_no_motif = np.sqrt(
                total_center_ss_no_motif[:, None] * total_neighbor_ss_no_motif[None, :]
            )
            corr_matrix_no_motif = total_cov_sum_no_motif / (denominator_no_motif + 1e-10)
            n_eff_no_motif = total_n_eff_no_motif  # Use accumulated n_eff
            print(f"Corr3 matrix shape: {corr_matrix_no_motif.shape}, effective n={n_eff_no_motif}")
        else:
            corr_matrix_no_motif = None
            n_eff_no_motif = 0

        # ====================================================================================
        # Step 4: Statistical testing
        # ====================================================================================
        print("\n" + "="*80)
        print("Step 4: Performing Fisher Z-tests")
        print("="*80)

        # Test 1: Corr1 vs Corr2 (neighbor vs non-neighbor)
        if total_cov_sum_non is not None and n_eff_non_neighbor > 0:
            _, p_value_test1 = spatial_utils.fisher_z_test(
                corr_matrix_neighbor, n_eff_neighbor,
                corr_matrix_non_neighbor, n_eff_non_neighbor
            )
            delta_corr_test1 = corr_matrix_neighbor - corr_matrix_non_neighbor
            print(f"Test1 completed: neighbor vs non-neighbor")
        else:
            p_value_test1 = np.ones((n_genes, n_genes))
            delta_corr_test1 = np.zeros((n_genes, n_genes))
            print("Test1 skipped: no non-neighbor data")

        # Test 2: Corr1 vs Corr3 (with motif vs without motif)
        if corr_matrix_no_motif is not None and n_eff_no_motif > 0:
            _, p_value_test2 = spatial_utils.fisher_z_test(
                corr_matrix_neighbor, n_eff_neighbor,
                corr_matrix_no_motif, n_eff_no_motif
            )
            delta_corr_test2 = corr_matrix_neighbor - corr_matrix_no_motif

            # Combined score
            combined_score = (0.3 * delta_corr_test1 * (-np.log10(p_value_test1 + 1e-300)) +
                            0.7 * delta_corr_test2 * (-np.log10(p_value_test2 + 1e-300)))
            print(f"Test2 completed: with motif vs without motif")
        else:
            p_value_test2 = None
            delta_corr_test2 = None
            combined_score = None
            print("Test2 skipped: no no-motif data")

        # ====================================================================================
        # Step 5: Build results DataFrame
        # ====================================================================================
        print("\n" + "="*80)
        print("Step 5: Building results DataFrame")
        print("="*80)

        # Create meshgrid for gene pairs
        gene_center_idx, gene_motif_idx = np.meshgrid(np.arange(n_genes), np.arange(n_genes), indexing='ij')

        results_df = pd.DataFrame({
            'gene_center': np.array(genes)[gene_center_idx.flatten()],
            'gene_motif': np.array(genes)[gene_motif_idx.flatten()],
            'corr_neighbor': corr_matrix_neighbor.flatten(),
            'corr_non_neighbor': corr_matrix_non_neighbor.flatten(),
            'p_value_test1': p_value_test1.flatten(),
            'delta_corr_test1': delta_corr_test1.flatten(),
        })

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

        # FDR correction
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

                rejected, adj_p_values = multipletests(all_p_values, method='fdr_bh')[:2]

                adj_p_test1 = adj_p_values[:n_consistent]
                adj_p_test2 = adj_p_values[n_consistent:]

                results_df['adj_p_value_test1'] = np.nan
                results_df['adj_p_value_test2'] = np.nan
                results_df['if_significant'] = False

                results_df.loc[same_direction, 'adj_p_value_test1'] = adj_p_test1
                results_df.loc[same_direction, 'adj_p_value_test2'] = adj_p_test2

                sig_mask = same_direction.copy()
                sig_indices = np.where(same_direction)[0]
                sig_both = (adj_p_test1 < 0.05) & (adj_p_test2 < 0.05)
                sig_mask[sig_indices] = sig_both

                results_df.loc[sig_mask, 'if_significant'] = True

                print(f"Significant gene pairs (both tests, FDR < 0.05): {sig_mask.sum()}")
            else:
                results_df['adj_p_value_test1'] = np.nan
                results_df['adj_p_value_test2'] = np.nan
                results_df['if_significant'] = False
        else:
            # Only test1 available
            rejected, adj_p_values = multipletests(results_df['p_value_test1'], method='fdr_bh')[:2]
            results_df['adj_p_value_test1'] = adj_p_values
            results_df['adj_p_value_test2'] = np.nan
            results_df['if_significant'] = rejected
            print(f"Significant gene pairs (test1, FDR < 0.05): {rejected.sum()}")

        # Sort by significance
        if corr_matrix_no_motif is not None and 'combined_score' in results_df.columns:
            results_df = results_df.sort_values('combined_score', ascending=False, ignore_index=True)
        else:
            results_df = results_df.sort_values('adj_p_value_test1', ignore_index=True)

       

        print("\n" + "="*80)
        print("Analysis completed!")
        print("="*80)
        print(f"Analyzed {len(selected_queries)} FOVs")
        print(f"Total gene pairs analyzed: {len(results_df)}")
        print(f"Significant pairs: {results_df['if_significant'].sum()}")

        return results_df

    def compute_gene_gene_correlation_by_type(self,
                                              ct: str,
                                              motif: Union[str, List[str]],
                                              dataset: Union[str, List[str]] = None,
                                              genes: Optional[Union[str, List[str]]] = None,
                                              max_dist: Optional[float] = None,
                                              k: Optional[int] = None,
                                              min_size: int = 0,
                                              min_nonzero: int = 10,
                                              ) -> pd.DataFrame:
        """
        Compute gene-gene cross correlation separately for each cell type in the motif across multiple FOVs.

        Similar to compute_gene_gene_correlation_by_type in spatial_query.py, but aggregates across FOVs.
        For each non-center cell type in the motif, compute:
        - Correlation 1: Center cells with motif vs neighboring motif cells of THIS TYPE
        - Correlation 2: Center cells with motif vs distant motif cells of THIS TYPE
        - Correlation 3: Center cells without motif vs neighbors (same for all types)

        Parameters
        ----------
        ct : str
            Cell type as the center cells.
        motif : Union[str, List[str]]
            Motif (names of cell types) to be analyzed. Include all cell types for neighbor finding.
        dataset : Union[str, List[str]], optional
            Datasets to include in analysis. If None, use all datasets.
        genes : Optional[Union[str, List[str]]], optional
            List of genes to analyze. If None, uses intersection of genes across all FOVs.
        max_dist : Optional[float], optional
            Maximum distance for considering a cell as a neighbor. Use either max_dist or k.
        k : Optional[int], optional
            Number of nearest neighbors. Use either max_dist or k.
        min_size : int, default=0
            Minimum neighborhood size for each center cell (only used when max_dist is specified).
        min_nonzero : int, default=10
            Minimum number of non-zero expression values required for a gene to be included.

        Returns
        -------
        pd.DataFrame
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
        """

        motif = motif if isinstance(motif, list) else [motif]

        # Get non-center cell types in motif
        non_center_types = [m for m in motif if m != ct]

        if len(non_center_types) == 1:
            print(f"Only one non-center cell type in motif: {non_center_types}. Using compute_gene_gene_correlation method.")
            result, _ = self.compute_gene_gene_correlation(
                ct=ct,
                motif=motif,
                dataset=dataset,
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

        # Select FOVs (handle dataset names with and without suffix)
        if dataset is None:
            dataset = [s.dataset.split('_')[0] for s in self.queries]
            print(f"No dataset specified. Using all datasets.")
        if isinstance(dataset, str):
            dataset = [dataset]

        valid_ds_names = [s.dataset.split('_')[0] for s in self.queries]
        for ds in dataset:
            if ds not in valid_ds_names:
                raise ValueError(f"Invalid input dataset name: {ds}.\n "
                               f"Valid dataset names are: {set(valid_ds_names)}")

        # Filter queries to include only selected datasets
        selected_queries = [s for s in self.queries if s.dataset.split('_')[0] in dataset]

        if len(selected_queries) == 0:
            raise ValueError(f"No FOVs found for dataset: {dataset}")

        print(f"Selected {len(selected_queries)} FOVs for analysis")

        # Check if ct and motif exist in at least one FOV
        ct_exists = any(ct in s.labels.unique() for s in selected_queries)
        if not ct_exists:
            raise ValueError(f"Center type '{ct}' not found in any selected datasets!")

        motif_exists = all(any(m in s.labels.unique() for s in selected_queries) for m in motif)
        if not motif_exists:
            missing = [m for m in motif if not any(m in s.labels.unique() for s in selected_queries)]
            raise ValueError(f"Motif types {missing} not found in any selected datasets!")

        # Get intersection of genes across all FOVs
        genes_sets = [set(sq.genes) for sq in selected_queries]
        all_genes = list(set.intersection(*genes_sets))
        if genes is None:
            print(f"No genes specified. Using all common genes across all selected FOVs ...")
            valid_genes = all_genes
        elif isinstance(genes, str):
            genes = [genes]
            valid_genes = [g for g in genes if g in all_genes]
        else:
            valid_genes = [g for g in genes if g in all_genes]

        if len(valid_genes) == 0:
            raise ValueError("No valid genes found across all FOVs.")

        genes = valid_genes
        n_genes = len(genes)
        print(f"Analyzing {n_genes} genes across {len(selected_queries)} FOVs")

        # ====================================================================================
        # Step 1: Accumulate Correlation 3 statistics (same for all types) across FOVs
        # ====================================================================================
        print("\n" + "="*80)
        print("Step 1: Computing Correlation-3 (Center without motif vs Neighbors)")
        print("="*80)

        total_cov_sum_no_motif = np.zeros((n_genes, n_genes))
        total_center_ss_no_motif = np.zeros(n_genes)
        total_neighbor_ss_no_motif = np.zeros(n_genes)
        total_n_pairs_no_motif = 0
        total_n_eff_no_motif = 0
        n_fovs_no_motif = 0

        # Also collect center_neighbor_pairs and non_neighbor_cells for each FOV
        fov_pair_data = []

        for fov_idx, sq in enumerate(selected_queries):
            print(f"\n--- Processing FOV {fov_idx + 1}/{len(selected_queries)}: {sq.dataset} ---")

            # Check if ct and motif types exist
            if ct not in sq.labels.unique():
                print(f"  Skipping: center type '{ct}' not in this FOV")
                continue

            missing_motif = [m for m in motif if m not in sq.labels.unique()]
            if missing_motif:
                print(f"  Skipping: motif types {missing_motif} not in this FOV")
                continue

            # Get expression data
            expr_genes = sq.adata[:, genes].X
            is_sparse = sparse.issparse(expr_genes)

            # Filter genes by non-zero expression
            if is_sparse:
                nonzero_fov = np.array((expr_genes > 0).sum(axis=0)).flatten()
            else:
                nonzero_fov = (expr_genes > 0).sum(axis=0)

            valid_gene_mask = nonzero_fov >= min_nonzero
            if valid_gene_mask.sum() < 10:
                print(f"  Skipping: only {valid_gene_mask.sum()} genes pass min_nonzero filter")
                continue

            # Compute FOV-specific cell type means
            fov_cell_type_means = {}
            for cell_type in sq.labels.unique():
                ct_mask = sq.labels == cell_type
                ct_cells = np.where(ct_mask)[0]
                if len(ct_cells) > 0:
                    ct_expr = expr_genes[ct_cells, :]
                    if is_sparse:
                        fov_cell_type_means[cell_type] = np.array(ct_expr.mean(axis=0)).flatten()
                    else:
                        fov_cell_type_means[cell_type] = ct_expr.mean(axis=0)

            center_mean = fov_cell_type_means[ct]

            # Get motif neighbor pairs
            try:
                neighbor_result = spatial_utils.get_motif_neighbor_cells(
                    sq_obj=sq, ct=ct, motif=motif, max_dist=max_dist, k=k, min_size=min_size
                )
                center_neighbor_pairs = neighbor_result['center_neighbor_pairs']
                ct_in_motif = neighbor_result['ct_in_motif']

                if len(center_neighbor_pairs) < 10:
                    print(f"  Skipping: only {len(center_neighbor_pairs)} pairs found")
                    continue

                # Get unique cells
                center_cells = np.unique(center_neighbor_pairs[:, 0])
                neighbor_cells = np.unique(center_neighbor_pairs[:, 1])

                # Get non-neighbor cells
                motif_mask = np.isin(sq.labels, motif)
                all_motif_cells = np.where(motif_mask)[0]
                non_neighbor_cells = np.setdiff1d(all_motif_cells, neighbor_cells)

                if ct_in_motif:
                    non_neighbor_cells = non_neighbor_cells[sq.labels[non_neighbor_cells] != ct]

                # Store pair data for this FOV (for Step 2)
                fov_pair_data.append({
                    'fov_idx': fov_idx,
                    'sq': sq,
                    'expr_genes': expr_genes,
                    'is_sparse': is_sparse,
                    'fov_cell_type_means': fov_cell_type_means,
                    'center_mean': center_mean,
                    'center_neighbor_pairs': center_neighbor_pairs,
                    'non_neighbor_cells': non_neighbor_cells,
                })

                # Compute Correlation 3
                no_motif_result = spatial_utils.get_all_neighbor_cells(
                    sq_obj=sq,
                    ct=ct,
                    max_dist=max_dist,
                    k=k,
                    min_size=min_size,
                    exclude_centers=center_cells,
                    exclude_neighbors=neighbor_cells,
                )

                center_no_motif_pairs = no_motif_result['center_neighbor_pairs']

                if len(center_no_motif_pairs) >= 10:
                    print(f"  Corr3: {len(center_no_motif_pairs)} no-motif pairs")

                    pair_centers_no_motif = center_no_motif_pairs[:, 0]
                    pair_neighbors_no_motif = center_no_motif_pairs[:, 1]
                    neighbor_no_motif_types = sq.labels[pair_neighbors_no_motif]

                    cov_sum, center_ss, neighbor_ss, n_pairs, n_eff = spatial_utils.compute_covariance_statistics_paired(
                        expr_genes=expr_genes,
                        pair_centers=pair_centers_no_motif,
                        pair_neighbors=pair_neighbors_no_motif,
                        center_mean=center_mean,
                        cell_type_means=fov_cell_type_means,
                        neighbor_cell_types=neighbor_no_motif_types,
                        is_sparse=is_sparse
                    )

                    total_cov_sum_no_motif += cov_sum
                    total_center_ss_no_motif += center_ss
                    total_neighbor_ss_no_motif += neighbor_ss
                    total_n_pairs_no_motif += n_pairs
                    total_n_eff_no_motif += n_eff
                    n_fovs_no_motif += 1

                    # Check for overflow/invalid values
                    if np.isinf(total_cov_sum_no_motif).any() or np.isnan(total_cov_sum_no_motif).any():
                        raise ValueError(f"Overflow or NaN detected in no-motif covariance sum at FOV {fov_idx + 1}")
                    if np.isinf(total_center_ss_no_motif).any() or np.isnan(total_center_ss_no_motif).any():
                        raise ValueError(f"Overflow or NaN detected in no-motif center_ss at FOV {fov_idx + 1}")
                    if np.isinf(total_neighbor_ss_no_motif).any() or np.isnan(total_neighbor_ss_no_motif).any():
                        raise ValueError(f"Overflow or NaN detected in no-motif neighbor_ss at FOV {fov_idx + 1}")
                else:
                    print(f"  Skipping Corr3: only {len(center_no_motif_pairs)} no-motif pairs")

            except Exception as e:
                print(f"  Error processing FOV: {e}")
                continue

        # Compute Correlation 3 matrix
        if n_fovs_no_motif > 0:
            denominator_no_motif = np.sqrt(
                total_center_ss_no_motif[:, None] * total_neighbor_ss_no_motif[None, :]
            )
            corr_matrix_no_motif = total_cov_sum_no_motif / (denominator_no_motif + 1e-10)
            n_eff_no_motif = total_n_eff_no_motif
            print(f"\nCorrelation 3: {total_n_pairs_no_motif} total pairs, n_eff={n_eff_no_motif} from {n_fovs_no_motif} FOVs")
        else:
            corr_matrix_no_motif = None
            n_eff_no_motif = 0
            print("\nWarning: No no-motif pairs found across any FOV!")

        # ====================================================================================
        # Step 2: Process each cell type separately
        # ====================================================================================
        print("\n" + "="*80)
        print("Step 2: Computing correlations for each cell type")
        print("="*80)

        all_results = []

        for cell_type in non_center_types:
            print(f"\n{'='*80}")
            print(f"Processing cell type: {cell_type}")
            print(f"{'='*80}")

            # Initialize accumulators for this cell type
            total_cov_sum_neighbor = np.zeros((n_genes, n_genes))
            total_center_ss_neighbor = np.zeros(n_genes)
            total_neighbor_ss_neighbor = np.zeros(n_genes)
            total_n_pairs_neighbor = 0
            total_n_eff_neighbor = 0

            total_cov_sum_non = np.zeros((n_genes, n_genes))
            total_center_ss_non = np.zeros(n_genes)
            total_non_neighbor_ss = np.zeros(n_genes)
            total_n_pairs_non = 0
            total_n_eff_non = 0

            n_fovs_this_type = 0

            # Process each FOV for this cell type
            for fov_data in fov_pair_data:
                sq = fov_data['sq']
                expr_genes = fov_data['expr_genes']
                is_sparse = fov_data['is_sparse']
                fov_cell_type_means = fov_data['fov_cell_type_means']
                center_mean = fov_data['center_mean']
                center_neighbor_pairs = fov_data['center_neighbor_pairs']
                non_neighbor_cells = fov_data['non_neighbor_cells']

                # Filter pairs for this cell type
                pair_neighbors = center_neighbor_pairs[:, 1]
                neighbor_types = sq.labels[pair_neighbors]
                type_mask = neighbor_types == cell_type

                if type_mask.sum() == 0:
                    continue

                type_specific_pairs = center_neighbor_pairs[type_mask]
                type_specific_center_cells = np.unique(type_specific_pairs[:, 0])

                # Filter non-neighbor cells for this type
                type_non_neighbor_mask = sq.labels[non_neighbor_cells] == cell_type
                type_non_neighbor_cells = non_neighbor_cells[type_non_neighbor_mask]

                if len(type_non_neighbor_cells) < 10:
                    continue

                n_fovs_this_type += 1

                # Correlation 1: neighboring cells of this type
                pair_centers = type_specific_pairs[:, 0]
                pair_neighbors_idx = type_specific_pairs[:, 1]
                neighbor_types_uniform = np.full(len(pair_neighbors_idx), cell_type)

                cov_sum, center_ss, neighbor_ss, n_pairs, n_eff = spatial_utils.compute_covariance_statistics_paired(
                    expr_genes=expr_genes,
                    pair_centers=pair_centers,
                    pair_neighbors=pair_neighbors_idx,
                    center_mean=center_mean,
                    cell_type_means=fov_cell_type_means,
                    neighbor_cell_types=neighbor_types_uniform,
                    is_sparse=is_sparse
                )

                total_cov_sum_neighbor += cov_sum
                total_center_ss_neighbor += center_ss
                total_neighbor_ss_neighbor += neighbor_ss
                total_n_pairs_neighbor += n_pairs
                total_n_eff_neighbor += n_eff

                # Check for overflow/invalid values
                if np.isinf(total_cov_sum_neighbor).any() or np.isnan(total_cov_sum_neighbor).any():
                    raise ValueError(f"Overflow or NaN detected in neighbor covariance sum for {cell_type}")
                if np.isinf(total_center_ss_neighbor).any() or np.isnan(total_center_ss_neighbor).any():
                    raise ValueError(f"Overflow or NaN detected in neighbor center_ss for {cell_type}")
                if np.isinf(total_neighbor_ss_neighbor).any() or np.isnan(total_neighbor_ss_neighbor).any():
                    raise ValueError(f"Overflow or NaN detected in neighbor neighbor_ss for {cell_type}")

                # Correlation 2: distant cells of this type
                non_neighbor_types_uniform = np.full(len(type_non_neighbor_cells), cell_type)

                cov_sum_non, center_ss_non, non_neighbor_ss, n_pairs_non, n_eff_non = spatial_utils.compute_covariance_statistics_all_to_all(
                    expr_genes=expr_genes,
                    center_cells=type_specific_center_cells,
                    neighbor_cells=type_non_neighbor_cells,
                    center_mean=center_mean,
                    cell_type_means=fov_cell_type_means,
                    neighbor_cell_types=non_neighbor_types_uniform,
                    is_sparse=is_sparse
                )

                total_cov_sum_non += cov_sum_non
                total_center_ss_non += center_ss_non
                total_non_neighbor_ss += non_neighbor_ss
                total_n_pairs_non += n_pairs_non
                total_n_eff_non += n_eff_non

                # Check for overflow/invalid values
                if np.isinf(total_cov_sum_non).any() or np.isnan(total_cov_sum_non).any():
                    raise ValueError(f"Overflow or NaN detected in non-neighbor covariance sum for {cell_type}")
                if np.isinf(total_center_ss_non).any() or np.isnan(total_center_ss_non).any():
                    raise ValueError(f"Overflow or NaN detected in non-neighbor center_ss for {cell_type}")
                if np.isinf(total_non_neighbor_ss).any() or np.isnan(total_non_neighbor_ss).any():
                    raise ValueError(f"Overflow or NaN detected in non-neighbor non_neighbor_ss for {cell_type}")

            if n_fovs_this_type == 0:
                print(f"No valid FOVs found for cell type {cell_type}. Skipping.")
                continue

            print(f"Processed {n_fovs_this_type} FOVs for {cell_type}")
            print(f"  Corr1: {total_n_pairs_neighbor} pairs, n_eff={total_n_eff_neighbor}")
            print(f"  Corr2: {total_n_pairs_non} pairs, n_eff={total_n_eff_non}")

            # Compute correlation matrices
            denominator_neighbor = np.sqrt(
                total_center_ss_neighbor[:, None] * total_neighbor_ss_neighbor[None, :]
            )
            corr_matrix_neighbor = total_cov_sum_neighbor / (denominator_neighbor + 1e-10)
            n_eff_neighbor = total_n_eff_neighbor

            denominator_non = np.sqrt(
                total_center_ss_non[:, None] * total_non_neighbor_ss[None, :]
            )
            corr_matrix_non_neighbor = total_cov_sum_non / (denominator_non + 1e-10)
            n_eff_non_neighbor = total_n_eff_non

            # Statistical testing
            print(f"\nPerforming statistical tests for {cell_type}...")

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
                'gene_center': np.array(genes)[gene_center_idx.flatten()],
                'gene_motif': np.array(genes)[gene_motif_idx.flatten()],
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

        # ====================================================================================
        # Step 3: Combine results and apply FDR correction
        # ====================================================================================
        print("\n" + "="*80)
        print("Step 3: Combining results and applying FDR correction")
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
                combined_results = combined_results[same_direction].copy()
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
                                       dataset: Union[str, List[str]] = None,
                                       genes: Optional[Union[str, List[str]]] = None,
                                       max_dist: Optional[float] = None,
                                       k: Optional[int] = None,
                                       min_size: int = 0,
                                       min_nonzero: int = 10,
                                       ) -> pd.DataFrame:
        """
        Compute gene-gene co-varying patterns using binary expression data from scfind index across multiple FOVs.

        Similar to compute_gene_gene_correlation in multiple FOV, but:
        - Uses binary expression data from scfind index instead of expression values
        - Aggregates center-neighbor pairs across all FOVs in specified dataset
        - Uses FOV-specific cell type means for centering (NOT global means)
        - Computes correlations by accumulating statistics across FOVs

        This function calculates cross correlation between gene expression in:
        1. Motif cells that are neighbors of center cell type (paired data across FOVs)
        2. Motif cells that are NOT neighbors of center cell type (all-to-all across FOVs)
        3. Neighboring cells of center cell type without nearby motif (paired data across FOVs)

        Parameter
        ---------
        ct:
            Cell type as the center cells.
        motif:
            Motif (names of cell types) to be analyzed.
        dataset:
            Datasets to include in analysis. If None, use all datasets.
        genes:
            List of genes to analyze. If None, uses intersection of genes across all FOVs.
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
        results_df : pd.DataFrame
            DataFrame with correlation results between neighbor and non-neighbor groups.
            Columns include:
                - gene_center, gene_motif: gene pairs
                - corr_neighbor: correlation in neighbor group
                - corr_non_neighbor: correlation in non-neighbor group
                - corr_center_no_motif: correlation for centers without motif
                - p_value_test1: p-value for test1 (neighbor vs non-neighbor)
                - p_value_test2: p-value for test2 (with motif vs without motif)
                - delta_corr_test1, delta_corr_test2: correlation differences
                - combined_score: combined significance score
                - adj_p_value_test1, adj_p_value_test2: FDR-corrected p-values
        """

        # Validate parameters
        if (max_dist is None and k is None) or (max_dist is not None and k is not None):
            raise ValueError("Please specify either max_dist or k, but not both.")

        # Convert motif to list
        motif = motif if isinstance(motif, list) else [motif]

        # Validate and prepare dataset list
        if dataset is None:
            dataset = [s.dataset.split('_')[0] for s in self.spatial_queries]
            print(f"No dataset specified. Using all datasets.")
        if isinstance(dataset, str):
            dataset = [dataset]

        valid_ds_names = [s.dataset.split('_')[0] for s in self.spatial_queries]
        for ds in dataset:
            if ds not in valid_ds_names:
                raise ValueError(f"Invalid input dataset name: {ds}.\n "
                               f"Valid dataset names are: {set(valid_ds_names)}")

        # Filter spatial_queries to include only selected datasets
        selected_queries = [s for s in self.spatial_queries if s.dataset.split('_')[0] in dataset]

        # Check if ct and motif exist in at least one FOV
        ct_exists = any(ct in s.labels.unique() for s in selected_queries)
        if not ct_exists:
            raise ValueError(f"Center type '{ct}' not found in any selected datasets!")

        motif_exists = all(any(m in s.labels.unique() for s in selected_queries) for m in motif)
        if not motif_exists:
            missing = [m for m in motif if not any(m in s.labels.unique() for s in selected_queries)]
            raise ValueError(f"Motif types {missing} not found in any selected datasets!")

        # Get intersection of genes across all FOVs
        genes_sets = [set(sq.genes) for sq in selected_queries]
        all_genes = list(set.intersection(*genes_sets))
        if genes is None:
            print(f"No genes specified. Using all common genes across all selected FOVs ...")
            valid_genes = all_genes
        elif isinstance(genes, str):
            genes = [genes]
            valid_genes = [g for g in genes if g in all_genes]
        else:
            valid_genes = [g for g in genes if g in all_genes]

        if len(valid_genes) == 0:
            raise ValueError("No valid genes found across all FOVs.")

        genes = valid_genes
        n_genes = len(genes)
        print(f"Analyzing {n_genes} genes across {len(selected_queries)} FOVs")

        # ====================================================================================
        # Step 1: FOV-level computation - collect statistics from each FOV
        # ====================================================================================
        print("\n" + "="*80)
        print("Step 1: Computing and accumulating statistics across FOVs using binary data")
        print("="*80)

        # Initialize accumulators for aggregated statistics
        # Neighbor (Correlation 1)
        total_cov_sum_neighbor = np.zeros((n_genes, n_genes))
        total_center_ss_neighbor = np.zeros(n_genes)
        total_neighbor_ss_neighbor = np.zeros(n_genes)
        total_n_pairs_neighbor = 0
        total_n_eff_neighbor = 0
        n_fovs_neighbor = 0

        # Non-neighbor (Correlation 2)
        total_cov_sum_non = np.zeros((n_genes, n_genes))
        total_center_ss_non = np.zeros(n_genes)
        total_non_neighbor_ss = np.zeros(n_genes)
        total_n_pairs_non = 0
        total_n_eff_non = 0
        n_fovs_non = 0

        # No-motif (Correlation 3)
        total_cov_sum_no_motif = np.zeros((n_genes, n_genes))
        total_center_ss_no_motif = np.zeros(n_genes)
        total_neighbor_ss_no_motif = np.zeros(n_genes)
        total_n_pairs_no_motif = 0
        total_n_eff_no_motif = 0
        n_fovs_no_motif = 0

        for fov_idx, sq in enumerate(selected_queries):
            print(f"\n--- Processing FOV {fov_idx + 1}/{len(selected_queries)}: {sq.dataset} ---")

            # Check if ct and all motif types exist in this FOV
            if ct not in sq.labels.unique():
                print(f"  Skipping: center type '{ct}' not in this FOV")
                continue

            missing_motif = [m for m in motif if m not in sq.labels.unique()]
            if missing_motif:
                print(f"  Skipping: motif types {missing_motif} not in this FOV")
                continue

            # Get binary expression data from scfind index for this FOV
            print(f"  Building binary expression matrix from scfind index...")
            sparse_data = sq.index.index.getBinarySparseMatrixData(valid_genes, sq.dataset, min_nonzero)

            rows = sparse_data['rows']
            cols = sparse_data['cols']
            filtered_genes = sparse_data['gene_names']
            n_cells = sparse_data['n_cells']

            if len(filtered_genes) == 0:
                print(f"  Skipping: no genes passed min_nonzero filter")
                continue

            # Create binary sparse matrix
            binary_expr = sparse.csr_matrix(
                (np.ones(len(rows), dtype=np.int16), (rows, cols)),
                shape=(n_cells, len(filtered_genes)),
            )
            is_sparse = True

            # Filter genes by non-zero expression in this FOV
            nonzero_fov = np.array((binary_expr > 0).sum(axis=0)).flatten()
            valid_gene_mask = nonzero_fov >= min_nonzero

            if valid_gene_mask.sum() < 10:
                print(f"  Skipping: only {valid_gene_mask.sum()} genes pass min_nonzero filter")
                continue

            # Compute FOV-specific cell type means for binary data
            fov_cell_type_means = {}
            for cell_type in sq.labels.unique():
                ct_mask = sq.labels == cell_type
                ct_cells = np.where(ct_mask)[0]
                if len(ct_cells) > 0:
                    ct_expr = binary_expr[ct_cells, :]
                    fov_cell_type_means[cell_type] = np.array(ct_expr.mean(axis=0)).flatten()

            center_mean = fov_cell_type_means[ct]

            # ========================================================================
            # Correlation 1: Center with motif vs Neighboring motif (PAIRED)
            # ========================================================================
            try:
                neighbor_result = spatial_utils.get_motif_neighbor_cells(
                    sq_obj=sq, ct=ct, motif=motif, max_dist=max_dist, k=k, min_size=min_size
                )
                center_neighbor_pairs = neighbor_result['center_neighbor_pairs']

                if len(center_neighbor_pairs) < 10:
                    print(f"  Skipping Corr1: only {len(center_neighbor_pairs)} pairs found")
                else:
                    print(f"  Corr1: {len(center_neighbor_pairs)} center-neighbor pairs")

                    # Extract pair indices
                    pair_centers = center_neighbor_pairs[:, 0]
                    pair_neighbors = center_neighbor_pairs[:, 1]

                    # Get neighbor cell types
                    neighbor_types = sq.labels[pair_neighbors]

                    # Compute statistics using optimized sparse-aware function
                    cov_sum, center_ss, neighbor_ss, n_pairs, n_eff = spatial_utils.compute_covariance_statistics_paired(
                        expr_genes=binary_expr,
                        pair_centers=pair_centers,
                        pair_neighbors=pair_neighbors,
                        center_mean=center_mean,
                        cell_type_means=fov_cell_type_means,
                        neighbor_cell_types=neighbor_types,
                        is_sparse=is_sparse
                    )

                    # Accumulate statistics directly (no intermediate storage)
                    total_cov_sum_neighbor += cov_sum
                    total_center_ss_neighbor += center_ss
                    total_neighbor_ss_neighbor += neighbor_ss
                    total_n_pairs_neighbor += n_pairs
                    total_n_eff_neighbor += n_eff
                    n_fovs_neighbor += 1

                    # Check for overflow/invalid values
                    if np.isinf(total_cov_sum_neighbor).any() or np.isnan(total_cov_sum_neighbor).any():
                        raise ValueError(f"Overflow or NaN detected in neighbor covariance sum at FOV {fov_idx + 1}")
                    if np.isinf(total_center_ss_neighbor).any() or np.isnan(total_center_ss_neighbor).any():
                        raise ValueError(f"Overflow or NaN detected in neighbor center_ss at FOV {fov_idx + 1}")
                    if np.isinf(total_neighbor_ss_neighbor).any() or np.isnan(total_neighbor_ss_neighbor).any():
                        raise ValueError(f"Overflow or NaN detected in neighbor neighbor_ss at FOV {fov_idx + 1}")

                    # ========================================================================
                    # Correlation 2: Center with motif vs Distant motif (ALL-TO-ALL)
                    # ========================================================================
                    # Get non-neighbor motif cells
                    motif_mask = np.isin(sq.labels.values, motif)
                    all_motif_cells = np.where(motif_mask)[0]
                    neighbor_cells_in_fov = np.unique(pair_neighbors)
                    non_neighbor_cells = np.setdiff1d(all_motif_cells, neighbor_cells_in_fov)

                    # Remove center type from non-neighbor
                    ct_in_motif = ct in motif
                    if ct_in_motif:
                        non_neighbor_cells = non_neighbor_cells[sq.labels[non_neighbor_cells] != ct]

                    if len(non_neighbor_cells) >= 10:
                        # Use unique center cells
                        unique_centers = np.unique(pair_centers)

                        print(f"  Corr2: {len(unique_centers)} centers × {len(non_neighbor_cells)} non-neighbors")

                        # Get non-neighbor types
                        non_neighbor_types = sq.labels[non_neighbor_cells]

                        # Compute statistics using optimized sparse-aware function
                        cov_sum_non, center_ss_non, non_neighbor_ss, n_pairs_non, n_eff_non = spatial_utils.compute_covariance_statistics_all_to_all(
                            expr_genes=binary_expr,
                            center_cells=unique_centers,
                            neighbor_cells=non_neighbor_cells,
                            center_mean=center_mean,
                            cell_type_means=fov_cell_type_means,
                            neighbor_cell_types=non_neighbor_types,
                            is_sparse=is_sparse
                        )

                        # Accumulate statistics directly
                        total_cov_sum_non += cov_sum_non
                        total_center_ss_non += center_ss_non
                        total_non_neighbor_ss += non_neighbor_ss
                        total_n_pairs_non += n_pairs_non
                        total_n_eff_non += n_eff_non
                        n_fovs_non += 1

                        # Check for overflow/invalid values
                        if np.isinf(total_cov_sum_non).any() or np.isnan(total_cov_sum_non).any():
                            raise ValueError(f"Overflow or NaN detected in non-neighbor covariance sum at FOV {fov_idx + 1}")
                        if np.isinf(total_center_ss_non).any() or np.isnan(total_center_ss_non).any():
                            raise ValueError(f"Overflow or NaN detected in non-neighbor center_ss at FOV {fov_idx + 1}")
                        if np.isinf(total_non_neighbor_ss).any() or np.isnan(total_non_neighbor_ss).any():
                            raise ValueError(f"Overflow or NaN detected in non-neighbor non_neighbor_ss at FOV {fov_idx + 1}")
                    else:
                        print(f"  Skipping Corr2: only {len(non_neighbor_cells)} non-neighbor cells")

                    # ========================================================================
                    # Correlation 3: Center without motif vs Neighbors (PAIRED)
                    # ========================================================================
                    no_motif_result = spatial_utils.get_all_neighbor_cells(
                        sq_obj=sq,
                        ct=ct,
                        max_dist=max_dist,
                        k=k,
                        min_size=min_size,
                        exclude_centers=np.unique(pair_centers),
                        exclude_neighbors=neighbor_cells_in_fov,
                    )

                    center_no_motif_pairs = no_motif_result['center_neighbor_pairs']

                    if len(center_no_motif_pairs) >= 10:
                        print(f"  Corr3: {len(center_no_motif_pairs)} no-motif pairs")

                        pair_centers_no_motif = center_no_motif_pairs[:, 0]
                        pair_neighbors_no_motif = center_no_motif_pairs[:, 1]

                        neighbor_no_motif_types = sq.labels[pair_neighbors_no_motif]

                        # Compute statistics using optimized sparse-aware function
                        cov_sum_no_motif, center_ss_no_motif, neighbor_ss_no_motif, n_pairs_no_motif, n_eff_no_motif = spatial_utils.compute_covariance_statistics_paired(
                            expr_genes=binary_expr,
                            pair_centers=pair_centers_no_motif,
                            pair_neighbors=pair_neighbors_no_motif,
                            center_mean=center_mean,
                            cell_type_means=fov_cell_type_means,
                            neighbor_cell_types=neighbor_no_motif_types,
                            is_sparse=is_sparse
                        )

                        # Accumulate statistics directly
                        total_cov_sum_no_motif += cov_sum_no_motif
                        total_center_ss_no_motif += center_ss_no_motif
                        total_neighbor_ss_no_motif += neighbor_ss_no_motif
                        total_n_pairs_no_motif += n_pairs_no_motif
                        total_n_eff_no_motif += n_eff_no_motif
                        n_fovs_no_motif += 1

                        # Check for overflow/invalid values
                        if np.isinf(total_cov_sum_no_motif).any() or np.isnan(total_cov_sum_no_motif).any():
                            raise ValueError(f"Overflow or NaN detected in no-motif covariance sum at FOV {fov_idx + 1}")
                        if np.isinf(total_center_ss_no_motif).any() or np.isnan(total_center_ss_no_motif).any():
                            raise ValueError(f"Overflow or NaN detected in no-motif center_ss at FOV {fov_idx + 1}")
                        if np.isinf(total_neighbor_ss_no_motif).any() or np.isnan(total_neighbor_ss_no_motif).any():
                            raise ValueError(f"Overflow or NaN detected in no-motif neighbor_ss at FOV {fov_idx + 1}")
                    else:
                        print(f"  Skipping Corr3: only {len(center_no_motif_pairs)} no-motif pairs")

            except Exception as e:
                print(f"  Error processing FOV: {e}")
                continue

        # ====================================================================================
        # Step 2: Validate and summarize accumulated statistics
        # ====================================================================================
        print("\n" + "="*80)
        print("Step 2: Summary of accumulated statistics")
        print("="*80)

        if n_fovs_neighbor == 0:
            raise ValueError("No valid neighbor pairs found across any FOV!")

        print(f"Correlation 1 (neighbor): {total_n_pairs_neighbor} total pairs, n_eff={total_n_eff_neighbor} from {n_fovs_neighbor} FOVs")

        if n_fovs_non > 0:
            print(f"Correlation 2 (non-neighbor): {total_n_pairs_non} total pairs, n_eff={total_n_eff_non} from {n_fovs_non} FOVs")
        else:
            print("Warning: No non-neighbor pairs found across any FOV!")
            total_cov_sum_non = None

        if n_fovs_no_motif > 0:
            print(f"Correlation 3 (no-motif): {total_n_pairs_no_motif} total pairs, n_eff={total_n_eff_no_motif} from {n_fovs_no_motif} FOVs")
        else:
            print("Warning: No no-motif pairs found across any FOV!")
            total_cov_sum_no_motif = None

        # ====================================================================================
        # Step 3: Compute correlation matrices
        # ====================================================================================
        print("\n" + "="*80)
        print("Step 3: Computing correlation matrices")
        print("="*80)

        # Correlation 1
        denominator_neighbor = np.sqrt(
            total_center_ss_neighbor[:, None] * total_neighbor_ss_neighbor[None, :]
        )
        corr_matrix_neighbor = total_cov_sum_neighbor / (denominator_neighbor + 1e-10)
        n_eff_neighbor = total_n_eff_neighbor  # Use accumulated n_eff

        print(f"Corr1 matrix shape: {corr_matrix_neighbor.shape}, effective n={n_eff_neighbor}")

        # Correlation 2
        if total_cov_sum_non is not None:
            denominator_non = np.sqrt(
                total_center_ss_non[:, None] * total_non_neighbor_ss[None, :]
            )
            corr_matrix_non_neighbor = total_cov_sum_non / (denominator_non + 1e-10)
            n_eff_non_neighbor = total_n_eff_non  # Use accumulated n_eff
            print(f"Corr2 matrix shape: {corr_matrix_non_neighbor.shape}, effective n={n_eff_non_neighbor}")
        else:
            corr_matrix_non_neighbor = np.zeros((n_genes, n_genes))
            n_eff_non_neighbor = 0

        # Correlation 3
        if total_cov_sum_no_motif is not None:
            denominator_no_motif = np.sqrt(
                total_center_ss_no_motif[:, None] * total_neighbor_ss_no_motif[None, :]
            )
            corr_matrix_no_motif = total_cov_sum_no_motif / (denominator_no_motif + 1e-10)
            n_eff_no_motif = total_n_eff_no_motif  # Use accumulated n_eff
            print(f"Corr3 matrix shape: {corr_matrix_no_motif.shape}, effective n={n_eff_no_motif}")
        else:
            corr_matrix_no_motif = None
            n_eff_no_motif = 0

        # ====================================================================================
        # Step 4: Statistical testing
        # ====================================================================================
        print("\n" + "="*80)
        print("Step 4: Performing Fisher Z-tests")
        print("="*80)

        # Test 1: Corr1 vs Corr2 (neighbor vs non-neighbor)
        if total_cov_sum_non is not None and n_eff_non_neighbor > 0:
            _, p_value_test1 = spatial_utils.fisher_z_test(
                corr_matrix_neighbor, n_eff_neighbor,
                corr_matrix_non_neighbor, n_eff_non_neighbor
            )
            delta_corr_test1 = corr_matrix_neighbor - corr_matrix_non_neighbor
            print(f"Test1 completed: neighbor vs non-neighbor")
        else:
            p_value_test1 = np.ones((n_genes, n_genes))
            delta_corr_test1 = np.zeros((n_genes, n_genes))
            print("Test1 skipped: no non-neighbor data")

        # Test 2: Corr1 vs Corr3 (with motif vs without motif)
        if corr_matrix_no_motif is not None and n_eff_no_motif > 0:
            _, p_value_test2 = spatial_utils.fisher_z_test(
                corr_matrix_neighbor, n_eff_neighbor,
                corr_matrix_no_motif, n_eff_no_motif
            )
            delta_corr_test2 = corr_matrix_neighbor - corr_matrix_no_motif

            # Combined score
            combined_score = (0.3 * delta_corr_test1 * (-np.log10(p_value_test1 + 1e-300)) +
                            0.7 * delta_corr_test2 * (-np.log10(p_value_test2 + 1e-300)))
            print(f"Test2 completed: with motif vs without motif")
        else:
            p_value_test2 = None
            delta_corr_test2 = None
            combined_score = None
            print("Test2 skipped: no no-motif data")

        # ====================================================================================
        # Step 5: Build results DataFrame
        # ====================================================================================
        print("\n" + "="*80)
        print("Step 5: Building results DataFrame")
        print("="*80)

        # Create meshgrid for gene pairs
        gene_center_idx, gene_motif_idx = np.meshgrid(np.arange(n_genes), np.arange(n_genes), indexing='ij')

        results_df = pd.DataFrame({
            'gene_center': np.array(genes)[gene_center_idx.flatten()],
            'gene_motif': np.array(genes)[gene_motif_idx.flatten()],
            'corr_neighbor': corr_matrix_neighbor.flatten(),
            'corr_non_neighbor': corr_matrix_non_neighbor.flatten(),
            'p_value_test1': p_value_test1.flatten(),
            'delta_corr_test1': delta_corr_test1.flatten(),
        })

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

        # FDR correction
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

                rejected, adj_p_values = multipletests(all_p_values, method='fdr_bh')[:2]

                adj_p_test1 = adj_p_values[:n_consistent]
                adj_p_test2 = adj_p_values[n_consistent:]

                results_df['adj_p_value_test1'] = np.nan
                results_df['adj_p_value_test2'] = np.nan
                results_df['if_significant'] = False

                results_df.loc[same_direction, 'adj_p_value_test1'] = adj_p_test1
                results_df.loc[same_direction, 'adj_p_value_test2'] = adj_p_test2

                sig_mask = same_direction.copy()
                sig_indices = np.where(same_direction)[0]
                sig_both = (adj_p_test1 < 0.05) & (adj_p_test2 < 0.05)
                sig_mask[sig_indices] = sig_both

                results_df.loc[sig_mask, 'if_significant'] = True

                print(f"Significant gene pairs (both tests, FDR < 0.05): {sig_mask.sum()}")
            else:
                results_df['adj_p_value_test1'] = np.nan
                results_df['adj_p_value_test2'] = np.nan
                results_df['if_significant'] = False
        else:
            # Only test1 available
            rejected, adj_p_values = multipletests(results_df['p_value_test1'], method='fdr_bh')[:2]
            results_df['adj_p_value_test1'] = adj_p_values
            results_df['adj_p_value_test2'] = np.nan
            results_df['if_significant'] = rejected
            print(f"Significant gene pairs (test1, FDR < 0.05): {rejected.sum()}")

        # Sort by significance
        if corr_matrix_no_motif is not None and 'combined_score' in results_df.columns:
            results_df = results_df.sort_values('combined_score', ascending=False, ignore_index=True)
        else:
            results_df = results_df.sort_values('adj_p_value_test1', ignore_index=True)

        print("\n" + "="*80)
        print("Analysis completed!")
        print("="*80)
        print(f"Analyzed {len(selected_queries)} FOVs")
        print(f"Total gene pairs analyzed: {len(results_df)}")
        print(f"Significant pairs: {results_df['if_significant'].sum()}")

        return results_df

    def compute_gene_gene_correlation_binary_by_type(self,
                                              ct: str,
                                              motif: Union[str, List[str]],
                                              dataset: Union[str, List[str]] = None,
                                              genes: Optional[Union[str, List[str]]] = None,
                                              max_dist: Optional[float] = None,
                                              k: Optional[int] = None,
                                              min_size: int = 0,
                                              min_nonzero: int = 10,
                                              ) -> pd.DataFrame:
        """
        Compute gene-gene cross correlation using binary expression data separately for each cell type in the motif across multiple FOVs.

        Similar to compute_gene_gene_correlation_by_type in multiple FOV, but uses binary expression data from scfind index.
        For each non-center cell type in the motif, compute:
        - Correlation 1: Center cells with motif vs neighboring motif cells of THIS TYPE
        - Correlation 2: Center cells with motif vs distant motif cells of THIS TYPE
        - Correlation 3: Center cells without motif vs neighbors (same for all types)

        Parameters
        ----------
        ct : str
            Cell type as the center cells.
        motif : Union[str, List[str]]
            Motif (names of cell types) to be analyzed. Include all cell types for neighbor finding.
        dataset : Union[str, List[str]], optional
            Datasets to include in analysis. If None, use all datasets.
        genes : Optional[Union[str, List[str]]], optional
            List of genes to analyze. If None, uses intersection of genes across all FOVs.
        max_dist : Optional[float], optional
            Maximum distance for considering a cell as a neighbor. Use either max_dist or k.
        k : Optional[int], optional
            Number of nearest neighbors. Use either max_dist or k.
        min_size : int, default=0
            Minimum neighborhood size for each center cell (only used when max_dist is specified).
        min_nonzero : int, default=10
            Minimum number of non-zero expression values required for a gene to be included.

        Returns
        -------
        pd.DataFrame
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
        """

        motif = motif if isinstance(motif, list) else [motif]

        # Get non-center cell types in motif
        non_center_types = [m for m in motif if m != ct]

        if len(non_center_types) == 1:
            print(f"Only one non-center cell type in motif: {non_center_types}. Using compute_gene_gene_correlation_binary method.")
            result = self.compute_gene_gene_correlation_binary(
                ct=ct,
                motif=motif,
                dataset=dataset,
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

        # Select FOVs (handle dataset names with and without suffix)
        if dataset is None:
            dataset = [s.dataset.split('_')[0] for s in self.spatial_queries]
            print(f"No dataset specified. Using all datasets.")
        if isinstance(dataset, str):
            dataset = [dataset]

        valid_ds_names = [s.dataset.split('_')[0] for s in self.spatial_queries]
        for ds in dataset:
            if ds not in valid_ds_names:
                raise ValueError(f"Invalid input dataset name: {ds}.\n "
                               f"Valid dataset names are: {set(valid_ds_names)}")

        # Filter queries to include only selected datasets
        selected_queries = [s for s in self.spatial_queries if s.dataset.split('_')[0] in dataset]

        if len(selected_queries) == 0:
            raise ValueError(f"No FOVs found for dataset: {dataset}")

        print(f"Selected {len(selected_queries)} FOVs for analysis")

        # Check if ct and motif exist in at least one FOV
        ct_exists = any(ct in s.labels.unique() for s in selected_queries)
        if not ct_exists:
            raise ValueError(f"Center type '{ct}' not found in any selected datasets!")

        motif_exists = all(any(m in s.labels.unique() for s in selected_queries) for m in motif)
        if not motif_exists:
            missing = [m for m in motif if not any(m in s.labels.unique() for s in selected_queries)]
            raise ValueError(f"Motif types {missing} not found in any selected datasets!")

        # Get intersection of genes across all FOVs
        genes_sets = [set(sq.genes) for sq in selected_queries]
        all_genes = list(set.intersection(*genes_sets))
        if genes is None:
            print(f"No genes specified. Using all common genes across all selected FOVs ...")
            valid_genes = all_genes
        elif isinstance(genes, str):
            genes = [genes]
            valid_genes = [g for g in genes if g in all_genes]
        else:
            valid_genes = [g for g in genes if g in all_genes]

        if len(valid_genes) == 0:
            raise ValueError("No valid genes found across all FOVs.")

        genes = valid_genes
        n_genes = len(genes)
        print(f"Analyzing {n_genes} genes across {len(selected_queries)} FOVs")

        # ====================================================================================
        # Step 1: Accumulate Correlation 3 statistics (same for all types) across FOVs
        # ====================================================================================
        print("\n" + "="*80)
        print("Step 1: Computing Correlation-3 (Center without motif vs Neighbors) using binary data")
        print("="*80)

        total_cov_sum_no_motif = np.zeros((n_genes, n_genes))
        total_center_ss_no_motif = np.zeros(n_genes)
        total_neighbor_ss_no_motif = np.zeros(n_genes)
        total_n_pairs_no_motif = 0
        total_n_eff_no_motif = 0
        n_fovs_no_motif = 0

        # Also collect center_neighbor_pairs and non_neighbor_cells for each FOV
        fov_pair_data = []

        for fov_idx, sq in enumerate(selected_queries):
            print(f"\n--- Processing FOV {fov_idx + 1}/{len(selected_queries)}: {sq.dataset} ---")

            # Check if ct and motif types exist
            if ct not in sq.labels.unique():
                print(f"  Skipping: center type '{ct}' not in this FOV")
                continue

            missing_motif = [m for m in motif if m not in sq.labels.unique()]
            if missing_motif:
                print(f"  Skipping: motif types {missing_motif} not in this FOV")
                continue

            # Get binary expression data from scfind index
            print(f"  Building binary expression matrix from scfind index...")
            sparse_data = sq.index.index.getBinarySparseMatrixData(valid_genes, sq.dataset, min_nonzero)

            rows = sparse_data['rows']
            cols = sparse_data['cols']
            filtered_genes = sparse_data['gene_names']
            n_cells = sparse_data['n_cells']

            if len(filtered_genes) == 0:
                print(f"  Skipping: no genes passed min_nonzero filter")
                continue

            # Create binary sparse matrix
            binary_expr = sparse.csr_matrix(
                (np.ones(len(rows), dtype=np.int16), (rows, cols)),
                shape=(n_cells, len(filtered_genes)),
            )
            is_sparse = True

            # Filter genes by non-zero expression
            nonzero_fov = np.array((binary_expr > 0).sum(axis=0)).flatten()
            valid_gene_mask = nonzero_fov >= min_nonzero
            if valid_gene_mask.sum() < 10:
                print(f"  Skipping: only {valid_gene_mask.sum()} genes pass min_nonzero filter")
                continue

            # Compute FOV-specific cell type means for binary data
            fov_cell_type_means = {}
            for cell_type in sq.labels.unique():
                ct_mask = sq.labels == cell_type
                ct_cells = np.where(ct_mask)[0]
                if len(ct_cells) > 0:
                    ct_expr = binary_expr[ct_cells, :]
                    fov_cell_type_means[cell_type] = np.array(ct_expr.mean(axis=0)).flatten()

            center_mean = fov_cell_type_means[ct]

            # Get motif neighbor pairs
            try:
                neighbor_result = spatial_utils.get_motif_neighbor_cells(
                    sq_obj=sq, ct=ct, motif=motif, max_dist=max_dist, k=k, min_size=min_size
                )
                center_neighbor_pairs = neighbor_result['center_neighbor_pairs']
                ct_in_motif = neighbor_result['ct_in_motif']

                if len(center_neighbor_pairs) < 10:
                    print(f"  Skipping: only {len(center_neighbor_pairs)} pairs found")
                    continue

                # Get unique cells
                center_cells = np.unique(center_neighbor_pairs[:, 0])
                neighbor_cells = np.unique(center_neighbor_pairs[:, 1])

                # Get non-neighbor cells
                motif_mask = np.isin(sq.labels, motif)
                all_motif_cells = np.where(motif_mask)[0]
                non_neighbor_cells = np.setdiff1d(all_motif_cells, neighbor_cells)

                if ct_in_motif:
                    non_neighbor_cells = non_neighbor_cells[sq.labels[non_neighbor_cells] != ct]

                # Store pair data for this FOV (for Step 2)
                fov_pair_data.append({
                    'fov_idx': fov_idx,
                    'sq': sq,
                    'expr_genes': binary_expr,
                    'is_sparse': is_sparse,
                    'fov_cell_type_means': fov_cell_type_means,
                    'center_mean': center_mean,
                    'center_neighbor_pairs': center_neighbor_pairs,
                    'non_neighbor_cells': non_neighbor_cells,
                })

                # Compute Correlation 3
                no_motif_result = spatial_utils.get_all_neighbor_cells(
                    sq_obj=sq,
                    ct=ct,
                    max_dist=max_dist,
                    k=k,
                    min_size=min_size,
                    exclude_centers=center_cells,
                    exclude_neighbors=neighbor_cells,
                )

                center_no_motif_pairs = no_motif_result['center_neighbor_pairs']

                if len(center_no_motif_pairs) >= 10:
                    print(f"  Corr3: {len(center_no_motif_pairs)} no-motif pairs")

                    pair_centers_no_motif = center_no_motif_pairs[:, 0]
                    pair_neighbors_no_motif = center_no_motif_pairs[:, 1]
                    neighbor_no_motif_types = sq.labels[pair_neighbors_no_motif]

                    cov_sum, center_ss, neighbor_ss, n_pairs, n_eff = spatial_utils.compute_covariance_statistics_paired(
                        expr_genes=binary_expr,
                        pair_centers=pair_centers_no_motif,
                        pair_neighbors=pair_neighbors_no_motif,
                        center_mean=center_mean,
                        cell_type_means=fov_cell_type_means,
                        neighbor_cell_types=neighbor_no_motif_types,
                        is_sparse=is_sparse
                    )

                    total_cov_sum_no_motif += cov_sum
                    total_center_ss_no_motif += center_ss
                    total_neighbor_ss_no_motif += neighbor_ss
                    total_n_pairs_no_motif += n_pairs
                    total_n_eff_no_motif += n_eff
                    n_fovs_no_motif += 1

                    # Check for overflow/invalid values
                    if np.isinf(total_cov_sum_no_motif).any() or np.isnan(total_cov_sum_no_motif).any():
                        raise ValueError(f"Overflow or NaN detected in no-motif covariance sum at FOV {fov_idx + 1}")
                    if np.isinf(total_center_ss_no_motif).any() or np.isnan(total_center_ss_no_motif).any():
                        raise ValueError(f"Overflow or NaN detected in no-motif center_ss at FOV {fov_idx + 1}")
                    if np.isinf(total_neighbor_ss_no_motif).any() or np.isnan(total_neighbor_ss_no_motif).any():
                        raise ValueError(f"Overflow or NaN detected in no-motif neighbor_ss at FOV {fov_idx + 1}")
                else:
                    print(f"  Skipping Corr3: only {len(center_no_motif_pairs)} no-motif pairs")

            except Exception as e:
                print(f"  Error processing FOV: {e}")
                continue

        # Compute Correlation 3 matrix
        if n_fovs_no_motif > 0:
            denominator_no_motif = np.sqrt(
                total_center_ss_no_motif[:, None] * total_neighbor_ss_no_motif[None, :]
            )
            corr_matrix_no_motif = total_cov_sum_no_motif / (denominator_no_motif + 1e-10)
            n_eff_no_motif = total_n_eff_no_motif
            print(f"\nCorrelation 3: {total_n_pairs_no_motif} total pairs, n_eff={n_eff_no_motif} from {n_fovs_no_motif} FOVs")
        else:
            corr_matrix_no_motif = None
            n_eff_no_motif = 0
            print("\nWarning: No no-motif pairs found across any FOV!")

        # ====================================================================================
        # Step 2: Process each cell type separately
        # ====================================================================================
        print("\n" + "="*80)
        print("Step 2: Computing correlations for each cell type")
        print("="*80)

        all_results = []

        for cell_type in non_center_types:
            print(f"\n{'='*80}")
            print(f"Processing cell type: {cell_type}")
            print(f"{'='*80}")

            # Initialize accumulators for this cell type
            total_cov_sum_neighbor = np.zeros((n_genes, n_genes))
            total_center_ss_neighbor = np.zeros(n_genes)
            total_neighbor_ss_neighbor = np.zeros(n_genes)
            total_n_pairs_neighbor = 0
            total_n_eff_neighbor = 0

            total_cov_sum_non = np.zeros((n_genes, n_genes))
            total_center_ss_non = np.zeros(n_genes)
            total_non_neighbor_ss = np.zeros(n_genes)
            total_n_pairs_non = 0
            total_n_eff_non = 0

            n_fovs_this_type = 0

            # Process each FOV for this cell type
            for fov_data in fov_pair_data:
                sq = fov_data['sq']
                expr_genes = fov_data['expr_genes']
                is_sparse = fov_data['is_sparse']
                fov_cell_type_means = fov_data['fov_cell_type_means']
                center_mean = fov_data['center_mean']
                center_neighbor_pairs = fov_data['center_neighbor_pairs']
                non_neighbor_cells = fov_data['non_neighbor_cells']

                # Filter pairs for this cell type
                pair_neighbors = center_neighbor_pairs[:, 1]
                neighbor_types = sq.labels[pair_neighbors]
                type_mask = neighbor_types == cell_type

                if type_mask.sum() == 0:
                    continue

                type_specific_pairs = center_neighbor_pairs[type_mask]
                type_specific_center_cells = np.unique(type_specific_pairs[:, 0])

                # Filter non-neighbor cells for this type
                type_non_neighbor_mask = sq.labels[non_neighbor_cells] == cell_type
                type_non_neighbor_cells = non_neighbor_cells[type_non_neighbor_mask]

                if len(type_non_neighbor_cells) < 10:
                    continue

                n_fovs_this_type += 1

                # Correlation 1: neighboring cells of this type
                pair_centers = type_specific_pairs[:, 0]
                pair_neighbors_idx = type_specific_pairs[:, 1]
                neighbor_types_uniform = np.full(len(pair_neighbors_idx), cell_type)

                cov_sum, center_ss, neighbor_ss, n_pairs, n_eff = spatial_utils.compute_covariance_statistics_paired(
                    expr_genes=expr_genes,
                    pair_centers=pair_centers,
                    pair_neighbors=pair_neighbors_idx,
                    center_mean=center_mean,
                    cell_type_means=fov_cell_type_means,
                    neighbor_cell_types=neighbor_types_uniform,
                    is_sparse=is_sparse
                )

                total_cov_sum_neighbor += cov_sum
                total_center_ss_neighbor += center_ss
                total_neighbor_ss_neighbor += neighbor_ss
                total_n_pairs_neighbor += n_pairs
                total_n_eff_neighbor += n_eff

                # Check for overflow/invalid values
                if np.isinf(total_cov_sum_neighbor).any() or np.isnan(total_cov_sum_neighbor).any():
                    raise ValueError(f"Overflow or NaN detected in neighbor covariance sum for {cell_type}")
                if np.isinf(total_center_ss_neighbor).any() or np.isnan(total_center_ss_neighbor).any():
                    raise ValueError(f"Overflow or NaN detected in neighbor center_ss for {cell_type}")
                if np.isinf(total_neighbor_ss_neighbor).any() or np.isnan(total_neighbor_ss_neighbor).any():
                    raise ValueError(f"Overflow or NaN detected in neighbor neighbor_ss for {cell_type}")

                # Correlation 2: distant cells of this type
                non_neighbor_types_uniform = np.full(len(type_non_neighbor_cells), cell_type)

                cov_sum_non, center_ss_non, non_neighbor_ss, n_pairs_non, n_eff_non = spatial_utils.compute_covariance_statistics_all_to_all(
                    expr_genes=expr_genes,
                    center_cells=type_specific_center_cells,
                    neighbor_cells=type_non_neighbor_cells,
                    center_mean=center_mean,
                    cell_type_means=fov_cell_type_means,
                    neighbor_cell_types=non_neighbor_types_uniform,
                    is_sparse=is_sparse
                )

                total_cov_sum_non += cov_sum_non
                total_center_ss_non += center_ss_non
                total_non_neighbor_ss += non_neighbor_ss
                total_n_pairs_non += n_pairs_non
                total_n_eff_non += n_eff_non

                # Check for overflow/invalid values
                if np.isinf(total_cov_sum_non).any() or np.isnan(total_cov_sum_non).any():
                    raise ValueError(f"Overflow or NaN detected in non-neighbor covariance sum for {cell_type}")
                if np.isinf(total_center_ss_non).any() or np.isnan(total_center_ss_non).any():
                    raise ValueError(f"Overflow or NaN detected in non-neighbor center_ss for {cell_type}")
                if np.isinf(total_non_neighbor_ss).any() or np.isnan(total_non_neighbor_ss).any():
                    raise ValueError(f"Overflow or NaN detected in non-neighbor non_neighbor_ss for {cell_type}")

            if n_fovs_this_type == 0:
                print(f"No valid FOVs found for cell type {cell_type}. Skipping.")
                continue

            print(f"Processed {n_fovs_this_type} FOVs for {cell_type}")
            print(f"  Corr1: {total_n_pairs_neighbor} pairs, n_eff={total_n_eff_neighbor}")
            print(f"  Corr2: {total_n_pairs_non} pairs, n_eff={total_n_eff_non}")

            # Compute correlation matrices
            denominator_neighbor = np.sqrt(
                total_center_ss_neighbor[:, None] * total_neighbor_ss_neighbor[None, :]
            )
            corr_matrix_neighbor = total_cov_sum_neighbor / (denominator_neighbor + 1e-10)
            n_eff_neighbor = total_n_eff_neighbor

            denominator_non = np.sqrt(
                total_center_ss_non[:, None] * total_non_neighbor_ss[None, :]
            )
            corr_matrix_non_neighbor = total_cov_sum_non / (denominator_non + 1e-10)
            n_eff_non_neighbor = total_n_eff_non

            # Statistical testing
            print(f"\nPerforming statistical tests for {cell_type}...")

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
                'gene_center': np.array(genes)[gene_center_idx.flatten()],
                'gene_motif': np.array(genes)[gene_motif_idx.flatten()],
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

        # ====================================================================================
        # Step 3: Combine results and apply FDR correction
        # ====================================================================================
        print("\n" + "="*80)
        print("Step 3: Combining results and applying FDR correction")
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
                combined_results = combined_results[same_direction].copy()
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
    