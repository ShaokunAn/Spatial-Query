from typing import List, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multitest as mt
from anndata import AnnData
from mlxtend.frequent_patterns import fpgrowth
from sklearn.preprocessing import MultiLabelBinarizer
from pandas import DataFrame
from scipy.stats import hypergeom
import time
from sklearn.preprocessing import LabelEncoder

from .spatial_query import spatial_query


class spatial_query_multi:
    def __init__(self,
                 adatas: List[AnnData],
                 datasets: List[str],
                 spatial_key: str,
                 label_key: str,
                 leaf_size: int, 
                 max_radius: float = 500,
                 n_split: int = 10,
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
        """
        # Each element in self.spatial_queries stores a spatial_query object
        self.spatial_key = spatial_key
        self.label_key = label_key
        self.max_radius = max_radius
        # Modify dataset names by d_0, d_2, ... for duplicates in datasets
        count_dict = {}
        modified_datasets = []
        for dataset in datasets:
            if '_' in dataset:
                print(f"Warning: Misusage of underscore in '{dataset}'. Replacing with hyphen.")
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
        fp = spatial_query.find_maximal_patterns(fp=fp)
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
        fp = spatial_query.find_maximal_patterns(fp=fp)
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

        Return
        ------
        pd.Dataframe containing the cell type name, motifs, number of motifs nearby given cell type,
        number of spots of cell type, number of motifs in single FOV, p value of hypergeometric distribution.
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

        for motif in motifs:
            n_labels = 0
            n_ct = 0
            n_motif_labels = 0
            n_motif_ct = 0

            motif = list(motif) if not isinstance(motif, list) else motif
            sort_motif = sorted(motif)

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

                    n_motif_labels += np.sum(np.all(neighbor_counts[:, int_motifs] > 0, axis=1))

                    if ct in np.unique(labels):
                        int_ct = label_encoder.transform(np.array(ct, dtype=object, ndmin=1))
                        mask = int_labels == int_ct
                        n_motif_ct += np.sum(np.all(neighbor_counts[mask][:, int_motifs] > 0, axis=1))
                        n_ct += np.sum(mask)

                # for i in range(len(labels)):
                #     if spatial_query.has_motif(sort_motif, [labels[idx] for idx in idxs[i][1:]]):
                #         n_motif_labels += 1
                # n_labels += len(labels)

                # if ct not in labels.unique():
                #     continue
                # cinds = [i for i, l in enumerate(labels) if l == ct]
                #
                # for i in cinds:
                #     inds = [ind for ind, d in enumerate(dists[i]) if d < max_dist]
                #     if len(inds) > 1:
                #         if spatial_query.has_motif(sort_motif, [labels[idx] for idx in idxs[i][inds[1:]]]):
                #             n_motif_ct += 1

                # n_ct += len(cinds)

            if ct in motif:
                n_ct = round(n_ct / motif.count(ct))

            hyge = hypergeom(M=n_labels, n=n_ct, N=n_motif_labels)
            motif_out = {'center': ct, 'motifs': sort_motif, 'n_center_motif': n_motif_ct,
                         'n_center': n_ct, 'n_motif': n_motif_labels, 'expectation': hyge.mean(), 'p-values': hyge.sf(n_motif_ct)}
            out.append(motif_out)

        out_pd = pd.DataFrame(out)

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
                              dataset: Union[str, List[str]] = None,
                              max_dist: float = 100,
                              min_size: int = 0,
                              min_support: float = 0.5,
                              max_ns: int = 100) -> DataFrame:
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
        Returns
        -------
        Tuple containing counts and statistical measures.
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
        if motifs is None:
            fp = self.find_fp_dist(ct=ct, dataset=dataset, max_dist=max_dist, min_size=min_size,
                                   min_support=min_support, max_ns=max_ns)
            motifs = fp['itemsets'].tolist()
        else:
            if isinstance(motifs, str):
                motifs = [motifs]

            all_labels = pd.concat([s.labels for s in self.spatial_queries])
            labels_unique_all = set(all_labels.unique())
            motifs_exc = [m for m in motifs if m not in labels_unique_all]
            if len(motifs_exc) != 0:
                print(f"Found no {motifs_exc} in {dataset}! Ignoring them.")
            motifs = [m for m in motifs if m not in motifs_exc]
            if len(motifs) == 0:
                raise ValueError(f"All cell types in motifs are missed in {self.label_key}.")
            motifs = [motifs]

        for motif in motifs:
            n_labels = 0
            n_ct = 0
            n_motif_labels = 0
            n_motif_ct = 0

            motif = list(motif) if not isinstance(motif, list) else motif
            sort_motif = sorted(motif)

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
                    _, matching_cells_indices = s._query_pattern(motif)
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

                    n_motif_labels += np.sum(np.all(neighbor_matrix[:, int_motifs] > 0, axis=1))

                    if ct in np.unique(labels):
                        int_ct = label_encoder.transform(np.array(ct, dtype=object, ndmin=1))
                        mask = int_labels[matching_cells_indices] == int_ct
                        print(f"{np.sum(np.all(neighbor_matrix[mask][:, int_motifs] > 0, axis=1))} n_center_motif")
                        n_motif_ct += np.sum(np.all(neighbor_matrix[mask][:, int_motifs] > 0, axis=1))
                        n_ct += np.sum(s.labels == ct)

                # ~10s using C++ codes
                # idxs = idxs.tolist()
                # cinds = [i for i, label in enumerate(labels) if label == ct]
                # n_motif_ct_s, n_motif_labels_s = spatial_module_utils.search_motif_dist(
                #     motif, idxs, labels, cinds, max_ns
                # )
                # n_motif_ct += n_motif_ct_s
                # n_motif_labels += n_motif_labels_s
                # n_ct += len(cinds)

                # original codes, ~ minutes
                # for i in range(len(idxs)):
                #     e = min(len(idxs[i]), max_ns)
                #     if spatial_query.has_motif(sort_motif, [labels[idx] for idx in idxs[i][:e] if idx != i]):
                #         n_motif_labels += 1
                #
                # if ct not in labels.unique():
                #     continue
                #
                # cinds = [i for i, label in enumerate(labels) if label == ct]
                #
                # for i in cinds:
                #     e = min(len(idxs[i]), max_ns)
                #     if spatial_query.has_motif(sort_motif, [labels[idx] for idx in idxs[i][:e] if idx != i]):
                #         n_motif_ct += 1
                #
                # n_ct += len(cinds)

            if ct in motif:
                n_ct = round(n_ct / motif.count(ct))
            hyge = hypergeom(M=n_labels, n=n_ct, N=n_motif_labels)
            motif_out = {'center': ct, 'motifs': sort_motif, 'n_center_motif': n_motif_ct,
                         'n_center': n_ct, 'n_motif': n_motif_labels, 'expectation': hyge.mean(), 'p-values': hyge.sf(n_motif_ct)}
            out.append(motif_out)

        out_pd = pd.DataFrame(out)

        p_values = out_pd['p-values'].tolist()
        if_rejected, corrected_p_values = mt.fdrcorrection(p_values,
                                                           alpha=0.05,
                                                           method='poscorr')
        out_pd['corrected p-values'] = corrected_p_values
        out_pd['if_significant'] = if_rejected
        out_pd = out_pd.sort_values(by='corrected p-values', ignore_index=True)
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
        fp, _, _ = sp_object.build_fptree_knn(
            cell_pos=ct_pos,
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

        fp, _, _ = sp_object.build_fptree_dist(cell_pos=ct_pos,
                                               max_dist=max_dist,
                                               min_support=min_support,
                                               min_size=min_size,
                                               if_max=False,
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
        fp_datasets['corrected_p_values'] = corrected_p_values
        fp_datasets['if_significant'] = if_rejected

        # Return the significant patterns in each dataset
        fp_dataset0 = fp_datasets[
            (fp_datasets['dataset_higher_frequency'] == datasets[0]) & (fp_datasets['if_significant'])
            ][['itemsets', 'corrected_p_values']]
        fp_dataset1 = fp_datasets[
            (fp_datasets['dataset_higher_frequency'] == datasets[1]) & (fp_datasets['if_significant'])
            ][['itemsets', 'corrected_p_values']]
        fp_dataset0 = fp_dataset0.reset_index(drop=True)
        fp_dataset1 = fp_dataset1.reset_index(drop=True)
        fp_dataset0 = fp_dataset0.sort_values(by='corrected_p_values', ascending=True)
        fp_dataset1 = fp_dataset1.sort_values(by='corrected_p_values', ascending=True)
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
        fp_datasets['corrected_p_values'] = corrected_p_values
        fp_datasets['if_significant'] = if_rejected

        # Return the significant patterns in each dataset
        fp_dataset0 = fp_datasets[
            (fp_datasets['dataset_higher_frequency'] == datasets[0]) & (fp_datasets['if_significant'])
            ][['itemsets', 'corrected_p_values']]
        fp_dataset1 = fp_datasets[
            (fp_datasets['dataset_higher_frequency'] == datasets[1]) & (fp_datasets['if_significant'])
            ][['itemsets', 'corrected_p_values']]

        fp_dataset0 = fp_dataset0.sort_values(by='corrected_p_values', ascending=True)
        fp_dataset1 = fp_dataset1.sort_values(by='corrected_p_values', ascending=True)
        fp_dataset0 = fp_dataset0.reset_index(drop=True)
        fp_dataset1 = fp_dataset1.reset_index(drop=True)
        return {datasets[0]: fp_dataset0, datasets[1]: fp_dataset1}
