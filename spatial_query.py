from pandas import DataFrame
from scipy.spatial import KDTree
from typing import List, Union
import numpy as np
from scipy.stats import hypergeom
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from anndata import AnnData
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import Counter, defaultdict


class spatial_query:
    def __init__(self,
                 adata: AnnData,
                 dataset: str = 'ST',
                 spatial_key: str = 'X_spatial',
                 label_key: str = 'predicted_label',
                 leaf_size: int = 10):
        if spatial_key not in adata.obsm.keys() or label_key not in adata.obs.keys():
            raise ValueError(f"The Anndata object must contain {spatial_key} in obsm and {label_key} in obs.")
        self.adata = adata
        self.dataset = dataset
        self.spatial_key = spatial_key
        self.label_key = label_key
        self.kd_tree = KDTree(self.adata.obsm[spatial_key], leafsize=leaf_size)

    @staticmethod
    def has_motif(neighbors: List[str], labels: List[str]) -> bool:
        """
        Determines whether all elements in 'neighbors' are present in 'labels'.
        If all elements are present, returns True. Otherwise, returns False.

        Parameter
        ---------
        neighbors:
            List of elements to check.
        labels:
            List in which to check for elements from 'neighbors'.

        Return
        ------
        True if all elements of 'neighbors' are in 'labels', False otherwise.
        """
        # Set elements in neighbors and labels to be unique.
        # neighbors = set(neighbors)
        # labels = set(labels)
        freq_neighbors = Counter(neighbors)
        freq_labels = Counter(labels)
        for element, count in freq_neighbors.items():
            if freq_labels[element] < count:
                return False

        return True
        # if len(neighbors) <= len(labels):
        #     for n in neighbors:
        #         if n in labels:
        #             pass
        #         else:
        #             return False
        #     return True
        # return False

    @staticmethod
    def _distinguish_duplicates(transaction: List[str]):
        """
        Append suffix to items of transaction to distinguish the duplicate items.
        """
        counter = dict(Counter(transaction))
        trans_suf = [f"{item}_{value}" for item, value in counter.items()]
        # count_dict = defaultdict(int)
        # for i, item in enumerate(transaction):
        #     # Increment the count for the item, or initialize it if it's new
        #     count_dict[item] += 1
        #     # Update the item with its count as suffix
        #     transaction[i] = f"{item}_{count_dict[item]}"
        # return transaction
        return trans_suf

    @staticmethod
    def find_maximal_patterns(fp: pd.DataFrame) -> pd.DataFrame:
        """
        Find the maximal frequent patterns

        Parameter
        ---------
            fp: Frequent patterns dataframe with support values and itemsets.

        Return
        ------
            Maximal frequent patterns with support and itemsets.
        """
        # Convert itemsets to frozensets for set operations
        itemsets = fp['itemsets'].apply(frozenset)

        # Find all subsets of each itemset
        subsets = set()
        for itemset in itemsets:
            for r in range(1, len(itemset)):
                subsets.update(frozenset(s) for s in combinations(itemset, r))

        # Identify maximal patterns (itemsets that are not subsets of any other)
        maximal_patterns = [itemset for itemset in itemsets if itemset not in subsets]

        # Filter the original DataFrame to keep only the maximal patterns
        return fp[fp['itemsets'].isin(maximal_patterns)].reset_index(drop=True)

    def find_fp_knn(self,
                    ct: str,
                    k: int = 20,
                    min_count: int = 0,
                    min_support: float = 0.5,
                    if_max: bool = True,
                    ) -> pd.DataFrame:
        """
        Find frequent patterns within the KNNs of certain cell type.

        Parameter
        ---------
        ct:
            Cell type name.
        k:
            Number of nearest neighbors.
        min_count:
            Minimum number of each cell type to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        if_max:
            Return all frequent patterns (if_max=False) or frequent patterns with maximal combinations (if_max=True).
            If a pattern (A, B, C) is frequent, its subsets (A, B), (A, C), (B, C) and (A), (B), (C) are also frequent.
            Return (A, B, C) if if_max=True otherwise return (A, B, C) and all its subsets.

        Return
        ------
        Frequent patterns in the neighborhood of certain cell type.
        """
        cell_pos = self.adata.obsm[self.spatial_key]
        labels = self.adata.obs[self.label_key]
        if ct not in labels.unique():
            raise ValueError(f"Not found {ct} in {self.label_key}!")

        cinds = [id for id, l in enumerate(labels) if l == ct]
        ct_pos = cell_pos[cinds]

        fp, _, _ = self.build_fptree_knn(cell_pos=ct_pos, k=k,
                                         min_count=min_count, min_support=min_support)
        if if_max:
            fp = self.find_maximal_patterns(fp=fp)

        return fp

    def find_fp_dist(self,
                     ct: str,
                     max_dist: float = 100,
                     min_size: int = 0,
                     min_count: int = 0,
                     min_support: float = 0.5,
                     if_max: bool = True,
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
        min_count:
            Minimum number of each cell type to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        if_max:
            Return all frequent patterns (if_max=False) or frequent patterns with maximal combinations (if_max=True).
            If a pattern (A, B, C) is frequent, its subsets (A, B), (A, C), (B, C) and (A), (B), (C) are also frequent.
            Return (A, B, C) if if_max=True otherwise return (A, B, C) and all its subsets.

        Return
        ------
        Frequent patterns in the neighborhood of certain cell type.
        """
        cell_pos = self.adata.obsm[self.spatial_key]
        labels = self.adata.obs[self.label_key]
        if ct not in labels.unique():
            raise ValueError(f"Not found {ct} in {self.label_key}!")

        cinds = [id for id, l in enumerate(labels) if l == ct]
        ct_pos = cell_pos[cinds]

        fp, _, _ = self.build_fptree_dist(cell_pos=ct_pos,
                                          max_dist=max_dist,
                                          min_size=min_size,
                                          min_count=min_count, min_support=min_support)
        if if_max:
            fp = self.find_maximal_patterns(fp=fp)

        return fp

    def motif_enrichment_knn(self,
                             ct: str,
                             motifs: Union[str, List[str]] = None,
                             k: int = 20,
                             min_count: int = 0,
                             min_support: float = 0.5,
                             if_max: bool = True,
                             max_dist: float = 200,
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
        min_count:
            Minimum number of each cell type to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        if_max:
            Return all frequent patterns (if_max=False) or frequent patterns with maximal combinations (if_max=True).
            If a pattern (A, B, C) is frequent, its subsets (A, B), (A, C), (B, C) and (A), (B), (C) are also frequent.
            Return (A, B, C) if if_max=True otherwise return (A, B, C) and all its subsets.
        max_dist:
            Maximum distance for neighbors (default: 200).

        Return
        ------
        pd.Dataframe containing the cell type name, motifs, number of motifs nearby given cell type,
        number of spots of cell type, number of motifs in single FOV, p value of hypergeometric distribution.
        """
        cell_pos = self.adata.obsm[self.spatial_key]
        labels = self.adata.obs[self.label_key]
        if ct not in labels.unique():
            raise ValueError(f"Not found {ct} in {self.label_key}!")

        dists, idxs = self.kd_tree.query(cell_pos, k=k + 1)  # use k+1 to find the knn except for the points themselves

        cinds = [i for i, l in enumerate(labels) if l == ct]

        out = []
        if motifs is None:
            fp = self.find_fp_knn(ct=ct, k=k, min_count=min_count,
                                  min_support=min_support, if_max=if_max)
            motifs = fp['itemsets']
        else:
            if isinstance(motifs, str):
                motifs = [motifs]

            labels_unique = labels.unique()
            motifs_exc = [m for m in motifs if m not in labels_unique]
            if len(motifs_exc) != 0:
                print(f"Found no {motifs_exc} in {self.label_key}. Ignoring them.")
            motifs = [m for m in motifs if m not in motifs_exc]
            motifs = [motifs]

        for motif in motifs:
            motif = list(motif) if not isinstance(motif, list) else motif
            sort_motif = sorted(motif)
            n_motif_ct = 0  # n_motif_ct is the number of centers nearby specified cell types (motif)
            for i in cinds:
                inds = [ind for ind, id in enumerate(dists[i]) if id < max_dist]
                if len(inds) > 1:
                    if self.has_motif(sort_motif, [labels[idx] for idx in idxs[i][inds[1:]]]):
                        n_motif_ct += 1

            n_motif_labels = 0  # n_motif_labels is the number of all cell_pos nearby specified motifs
            for i, _ in enumerate(labels):
                if self.has_motif(sort_motif, [labels[idx] for idx in idxs[i][1:]]):
                    n_motif_labels += 1

            n_ct = len(cinds)
            if ct in motif:
                n_ct = round(n_ct / motif.count(ct))

            hyge = hypergeom(M=len(labels), n=n_ct, N=n_motif_labels)
            # M is number of total, N is number of drawn without replacement, n is number of success in total
            motif_out = {'center': ct, 'motifs': sort_motif, 'n_center_motif': n_motif_ct,
                         'n_center': n_ct, 'n_motif': n_motif_labels, 'p-val': hyge.sf(n_motif_ct)}
            out.append(motif_out)

        out_pd = pd.DataFrame(out)
        out_pd = out_pd.sort_values(by='p-val', ignore_index=True)

        return out_pd

    def motif_enrichment_dist(self,
                              ct: str,
                              motifs: Union[str, List[str]] = None,
                              max_dist: float = 100,
                              min_size: int = 0,
                              min_count: int = 0,
                              min_support: float = 0.5,
                              if_max: bool = True,
                              max_ns: int = 1000000) -> DataFrame:
        """
        Perform motif enrichment analysis within a specified radius-based neighborhood.

        Parameter
        ---------
        ct:
            Cell type of the center cell.
        motifs:
            Specified motifs to be tested.
            If motifs=None, find the frequent patterns as motifs within the neighborhood of center cell type.
        max_dist:
            Maximum distance for considering a cell as a neighbor.
        min_size:
            Minimum neighborhood size for each point to consider.
        min_count:
            Minimum number of each cell type to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        if_max:
            Return all frequent patterns (if_max=False) or frequent patterns with maximal combinations (if_max=True).
            If a pattern (A, B, C) is frequent, its subsets (A, B), (A, C), (B, C) and (A), (B), (C) are also frequent.
            Return (A, B, C) if if_max=True otherwise return (A, B, C) and all its subsets.
        max_ns:
            Maximum number of neighborhood size for each point.
        Returns
        -------
        Tuple containing counts and statistical measures.
        """
        cell_pos = self.adata.obsm[self.spatial_key]
        labels = self.adata.obs[self.label_key]
        if ct not in labels.unique():
            raise ValueError(f"Not found {ct} in {self.label_key}!")

        idxs = self.kd_tree.query_ball_point(cell_pos, r=max_dist, return_sorted=True)
        cinds = [i for i, label in enumerate(labels) if label == ct]

        out = []
        if motifs is None:
            fp = self.find_fp_dist(ct=ct, max_dist=max_dist, min_size=min_size,
                                   min_count=min_count, min_support=min_support, if_max=if_max)
            motifs = fp['itemsets']
        else:
            if isinstance(motifs, str):
                motifs = [motifs]

            labels_unique = labels.unique()
            motifs_exc = [m for m in motifs if m not in labels_unique]
            if len(motifs_exc) != 0:
                print(f"Found no {motifs_exc} in {self.label_key}. Ignoring them.")
            motifs = [m for m in motifs if m not in motifs_exc]
            motifs = [motifs]

        for motif in motifs:
            motif = list(motif) if not isinstance(motif, list) else motif
            sort_motif = sorted(motif)
            n_motif_ct = 0
            for i in cinds:
                e = min(len(idxs[i]), max_ns)
                if self.has_motif(sort_motif, [labels[idx] for idx in idxs[i][1:e]]):
                    n_motif_ct += 1

            n_motif_labels = 0
            for i in range(len(idxs)):
                e = min(len(idxs[i]), max_ns)
                if self.has_motif(sort_motif, [labels[idx] for idx in idxs[i][1:e]]):
                    n_motif_labels += 1

            n_ct = len(cinds)
            if ct in motif:
                n_ct = round(n_ct / motif.count(ct))

            hyge = hypergeom(M=len(labels), n=n_ct, N=n_motif_labels)
            motif_out = {'center': ct, 'motifs': sort_motif, 'n_center_motif': n_motif_ct,
                         'n_center': n_ct, 'n_motif': n_motif_labels, 'p-val': hyge.sf(n_motif_ct)}
            out.append(motif_out)

        out_pd = pd.DataFrame(out)
        out_pd = out_pd.sort_values(by='p-val', ignore_index=True)
        return out_pd

    def build_fptree_dist(self,
                          cell_pos: np.ndarray = None,
                          max_dist: float = 100,
                          min_size: int = 0,
                          min_count: int = 0,
                          min_support: float = 0.5,
                          max_ns: int = 1000000) -> tuple:
        """
        Build a frequency pattern tree based on the distance of cell types.

        Parameter
        ---------
        cell_pos:
            Spatial coordinates of input points.
            If cell_pos is None, use all spots in fov to compute frequent patterns.
        max_dist:
            Maximum distance to consider a cell as a neighbor.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        min_size:
            Minimum neighborhood size for each point to consider.
        min_count:
            Minimum number of each cell type to consider.
        max_ns:
            Maximum number of neighborhood size for each point.

        Return
        ------
        A tuple containing the FPs, the transactions table and the nerghbors index.
        """
        if cell_pos is None:
            cell_pos = self.adata.obsm[self.spatial_key]

        labels = self.adata.obs[self.label_key]
        pos2neighbor_count = {}
        idxs = self.kd_tree.query_ball_point(cell_pos, r=max_dist, return_sorted=True)
        ct_all = sorted(set(labels))
        ct_count = np.zeros(len(ct_all), dtype=int)

        for i, idx in enumerate(idxs):
            if len(idx) > min_size + 1:
                for j in idx[1:min(max_ns, len(idx))]:
                    ct_count[ct_all.index(labels[j])] += 1
            pos2neighbor_count[tuple(cell_pos[i])] = len(idx)

        ct_exclude = [ct_all[i] for i, count in enumerate(ct_count) if count < min_count]

        # Prepare data for FP-Tree construction
        transactions = []
        valid_idxs = []
        for idx in idxs:
            transaction = [labels[i] for i in idx[1:min(max_ns, len(idx))] if labels[i] not in ct_exclude]
            # Append suffix to distinguish the duplicates in transaction
            if len(transaction) > min_size:
                transaction = self._distinguish_duplicates(transaction)
                transactions.append(transaction)
                valid_idxs.append(idx)

        # Convert transactions to a DataFrame suitable for fpgrowth
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        # Construct FP-Tree using fpgrowth
        fp_tree = fpgrowth(df, min_support=min_support, use_colnames=True)

        return fp_tree, df, valid_idxs

    def build_fptree_knn(self,
                         cell_pos: np.ndarray = None,
                         k: int = 20,
                         min_count: int = 0,
                         min_support: float = 0.5,
                         max_dist: float = 100,
                         ) -> tuple:
        """
        Build a frequency pattern tree based on knn

        Parameter
        ---------
        cell_pos:
            Spatial coordinates of input points.
            If cell_pos is None, use all spots in fov to compute frequent patterns.
        k:
            Number of neighborhood size for each point.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern
        min_count:
            Minimum number of cell type to consider.
        max_dist:
            The maximum distance at which points are considered neighbors.

        Return
        ------
        A tuple containing the FPs, the transactions table, and the nerghbors index.
        """
        if cell_pos is None:
            cell_pos = self.adata.obsm[self.spatial_key]

        labels = self.adata.obs[self.label_key]
        dists, idxs = self.kd_tree.query(cell_pos, k=k + 1)
        ct_all = sorted(set(labels))
        ct_count = np.zeros(len(ct_all), dtype=int)

        for i, idx in enumerate(idxs):
            for j in idx[1:]:
                ct_count[ct_all.index(labels[j])] += 1

        ct_exclude = [ct_all[i] for i, count in enumerate(ct_count) if count < min_count]

        # Prepare data for FP-Tree construction
        transactions = []
        for i, idx in enumerate(idxs):
            inds = [id for j, id in enumerate(idx) if
                    dists[i][j] < max_dist]  # only contain the KNN whose distance is less than max_dist
            transaction = [labels[i] for i in inds[1:] if labels[i] not in ct_exclude]
            transaction = self._distinguish_duplicates(transaction)
            transactions.append(transaction)

        # Convert transactions to a DataFrame suitable for fpgrowth
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        # Construct FP-Tree using fpgrowth
        fp_tree = fpgrowth(df, min_support=min_support, use_colnames=True)

        return fp_tree, df, idxs

    def find_patterns_grid(self,
                           max_dist: float = 100,
                           min_size: int = 0,
                           min_count: int = 0,
                           min_support: float = 0.5,
                           if_max: bool = False,
                           if_display: bool = True,
                           fig_size: tuple = (10, 5)
                           ) -> DataFrame:
        """
        Create a grid and use it to find surrounding patterns in spatial data.

        Parameter
        ---------
        max_dist:
            Maximum distance to consider a cell as a neighbor.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern
        min_size, min_count:
            Additional parameters for pattern finding.
        if_max:
            Return all frequent patterns (if_max=False) or frequent patterns with maximal combinations (if_max=True).
            If a pattern (A, B, C) is frequent, its subsets (A, B), (A, C), (B, C) and (A), (B), (C) are also frequent.
            Return (A, B, C) if if_max=True otherwise return (A, B, C) and all its subsets.
        if_display:
            Display the grid points with nearby frequent patterns if if_display=True.
        fig_size:
            Tuple of figure size.

        Return
        ------
        fp_tree:
            Frequent patterns
        """
        cell_pos = self.adata.obsm[self.spatial_key]
        labels = self.adata.obs[self.label_key]
        xmax, ymax = np.max(cell_pos, axis=0)
        xmin, ymin = np.min(cell_pos, axis=0)
        x_grid = np.arange(xmin - max_dist, xmax + max_dist, max_dist)
        y_grid = np.arange(ymin - max_dist, ymax + max_dist, max_dist)
        grid = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)

        fp, trans_df, idxs = self.build_fptree_dist(cell_pos=grid,
                                                    max_dist=max_dist, min_size=min_size,
                                                    min_count=min_count, min_support=min_support)
        if if_max:
            fp = self.find_maximal_patterns(fp=fp)

        if if_display:
            fp_cts = sorted(set(t for items in fp['itemsets'] for t in list(items)))
            n_colors = len(fp_cts)
            colors = sns.color_palette('hsv', n_colors)
            color_map = {ct: col for ct, col in zip(fp_cts, colors)}

            fp_spots_index = set()
            for motif in fp['itemsets']:
                motif = list(motif)
                ids = trans_df[motif].all(axis=1)
                if isinstance(idxs, list):
                    ids = ids.index[ids == True].to_list()
                    fp_spots_index.update([i for id in ids for i in idxs[id] if labels[i] in motif])
                else:
                    ids = idxs[ids]
                    fp_spots_index.update([i for id in ids for i in id if labels[i] in motif])

            fp_spot_pos = self.adata[list(fp_spots_index), :]
            fig, ax = plt.subplots(figsize=fig_size)
            for ct in fp_cts:
                ct_ind = fp_spot_pos.obs[self.label_key] == ct
                ax.scatter(fp_spot_pos.obsm[self.spatial_key][ct_ind, 0], fp_spot_pos.obsm[self.spatial_key][ct_ind, 1],
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

    def find_patterns_rand(self,
                           max_dist: float = 100,
                           n_points: int = 1000,
                           min_support: float = 0.5,
                           min_size: int = 0,
                           min_count: int = 0,
                           if_max: bool = False,
                           if_display: bool = True,
                           fig_size: tuple = (10, 5),
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
        min_size, min_count:
            Additional parameters for pattern finding.
        if_max:
            Return all frequent patterns (if_max=False) or frequent patterns with maximal combinations (if_max=True).
            If a pattern (A, B, C) is frequent, its subsets (A, B), (A, C), (B, C) and (A), (B), (C) are also frequent.
            Return (A, B, C) if if_max=True otherwise return (A, B, C) and all its subsets.
        if_display:
            Display the grid points with nearby frequent patterns if if_display=True.
        fig_size:
            Tuple of figure size.
        seed:
            Set random seed for reproducible.

        Return
        ------
        Results from the pattern finding function.
        """
        cell_pos = self.adata.obsm[self.spatial_key]
        labels = self.adata.obs[self.label_key]
        xmax, ymax = np.max(cell_pos, axis=0)
        xmin, ymin = np.min(cell_pos, axis=0)
        np.random.seed(seed)
        pos = np.column_stack((np.random.rand(n_points) * (xmax - xmin) + xmin,
                               np.random.rand(n_points) * (ymax - ymin) + ymin))

        fp, trans_df, idxs = self.build_fptree_dist(cell_pos=pos,
                                                    max_dist=max_dist, min_size=min_size,
                                                    min_count=min_count,
                                                    min_support=min_support)
        if if_max:
            fp = self.find_maximal_patterns(fp=fp)

        if if_display:
            fp_cts = sorted(set(t for items in fp['itemsets'] for t in list(items)))
            n_colors = len(fp_cts)
            colors = sns.color_palette('hsv', n_colors)
            color_map = {ct: col for ct, col in zip(fp_cts, colors)}

            fp_spots_index = set()
            for motif in fp['itemsets']:
                motif = list(motif)
                ids = trans_df[motif].all(axis=1)
                if isinstance(idxs, list):
                    ids = ids.index[ids == True].to_list()
                    fp_spots_index.update([i for id in ids for i in idxs[id] if labels[i] in motif])
                else:
                    ids = idxs[ids]
                    fp_spots_index.update([i for id in ids for i in id if labels[i] in motif])

            fp_spot_pos = self.adata[list(fp_spots_index), :]
            fig, ax = plt.subplots(figsize=fig_size)
            for ct in fp_cts:
                ct_ind = fp_spot_pos.obs[self.label_key] == ct
                ax.scatter(fp_spot_pos.obsm[self.spatial_key][ct_ind, 0], fp_spot_pos.obsm[self.spatial_key][ct_ind, 1],
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

    def plot_fov(self,
                 min_cells_label: int = 50,
                 title: str = 'Spatial distribution of cell types',
                 fig_size: tuple = (10, 5)):
        """
        Plot the cell type distribution of single fov.

        Parameter
        --------
        min_cells_label:
            Minimum number of points in each cell type to display.
        title:
            Figure title.
        fig_size:
            Figure size paramters.

        Return
        ------
        A figure.
        """
        # Ensure that 'spatial' and label_key are present in the Anndata object

        self.adata.obs[self.label_key] = self.adata.obs[self.label_key].astype('category')
        cell_type_counts = self.adata.obs[self.label_key].value_counts()
        n_colors = sum(cell_type_counts >= min_cells_label)
        colors = sns.color_palette('hsv', n_colors)

        color_counter = 0
        fig, ax = plt.subplots(figsize=fig_size)

        # Iterate over each cell type
        for cell_type in self.adata.obs[self.label_key].unique():
            # Filter data for each cell type
            data = self.adata.obs[self.adata.obs[self.label_key] == cell_type].index
            # Check if the cell type count is above the threshold
            if cell_type_counts[cell_type] >= min_cells_label:
                ax.scatter(self.adata[data].obsm[self.spatial_key][:, 0], self.adata[data].obsm[self.spatial_key][:, 1],
                           label=cell_type, color=colors[color_counter], s=1)
                color_counter += 1
            else:
                ax.scatter(self.adata[data].obsm[self.spatial_key][:, 0], self.adata[data].obsm[self.spatial_key][:, 1],
                           color='grey', s=1)

        handles, labels = ax.get_legend_handles_labels()

        # Modify labels to include count values
        new_labels = [f'{label} ({cell_type_counts[label]})' for label in labels]

        # Create new legend
        ax.legend(handles, new_labels, loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)

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

        plt.show()

    def plot_motif_grid(self,
                        motif: Union[str, List[str]],
                        max_dist: float = 100,
                        min_count: int = 0,
                        min_support: float = 0.5,
                        min_size: int = 0,
                        if_max: bool = True,
                        fig_size: tuple = (10, 5)
                        ):
        """
        Display the grid points with motif in radius-based neighborhood,
        and cell types of motif in the neighborhood of these grid points. To make sure the input
        motif can be found in the results obtained by find_patterns_grid, use the same arguments
        as those in find_pattern_grid method.

        Parameter
        ---------
        motif:
            Motif (names of cell types) to be colored
        max_dist:
            Spacing distance for building grid. Make sure using the same value as that in find_patterns_grid.
        min_count:
            Minimum number of each cell type to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        min_size:
            Minimum neighborhood size for each point to consider.
        if_max:
            Return all frequent patterns (if_max=False) or frequent patterns with maximal combinations (if_max=True).
            If a pattern (A, B, C) is frequent, its subsets (A, B), (A, C), (B, C) and (A), (B), (C) are also frequent.
            Return (A, B, C) if if_max=True otherwise return (A, B, C) and all its subsets.
        fig_size:
            Figure size.
        """
        if isinstance(motif, str):
            motif = [motif]

        cell_pos = self.adata.obsm[self.spatial_key]
        labels = self.adata.obs[self.label_key]
        labels_unique = labels.unique()
        motif_exc = [m for m in motif if m not in labels_unique]
        if len(motif_exc) != 0:
            print(f"Found no {motif_exc} in {self.label_key}. Ignoring them.")
        motif = [m for m in motif if m not in motif_exc]

        # Build mesh
        xmax, ymax = np.max(cell_pos, axis=0)
        xmin, ymin = np.min(cell_pos, axis=0)
        x_grid = np.arange(xmin - max_dist, xmax + max_dist, max_dist)
        y_grid = np.arange(ymin - max_dist, ymax + max_dist, max_dist)
        grid = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)

        # Compute fp here just to make sure we can use the same color map as in find_patterns_grid function.
        # If there's no need to keep same color map, can just use self.kd_tree.query() in knn or
        # self.kd_tree.query_ball_point in radisu-based neighborhood.
        fp, _, _ = self.build_fptree_dist(cell_pos=grid,
                                          max_dist=max_dist, min_size=min_size,
                                          min_count=min_count, min_support=min_support)
        # self.build_fptree_dist returns valid_idxs () instead of all the idxs,
        # so recalculate the idxs directly using self.kd_tree.query_ball_point
        idxs = self.kd_tree.query_ball_point(grid, r=max_dist, return_sorted=True)

        if if_max:
            fp = self.find_maximal_patterns(fp=fp)

        # Locate the index of grid points acting as centers with motif nearby
        id_center = []
        for i, idx in enumerate(idxs):
            ns = [labels[id] for id in idx[1:]]
            if self.has_motif(neighbors=motif, labels=ns):
                id_center.append(i)

        # Locate the index of cell types contained in motif in the
        # neighborhood of above grid points with motif nearby
        id_motif_celltype = set()  # the index of spots with cell types in motif and within the neighborhood of
        # above grid points
        for id in id_center:
            id_neighbor = [i for i in idxs[id][1:] if labels[i] in motif]
            id_motif_celltype.update(id_neighbor)

        # Plot above spots and center grid points
        # Set color map as in find_patterns_grid
        fp_cts = sorted(set(t for items in fp['itemsets'] for t in list(items)))
        n_colors = len(fp_cts)
        colors = sns.color_palette('hsv', n_colors)
        color_map = {ct: col for ct, col in zip(fp_cts, colors)}

        motif_spot_pos = self.adata[list(id_motif_celltype), :]
        fig, ax = plt.subplots(figsize=fig_size)
        ax.scatter(grid[id_center, 0], grid[id_center, 1], label='Grid Points',
                   edgecolors='red', facecolors='none', s=8)

        # Plotting the grid lines
        for x in x_grid:
            ax.axvline(x, color='lightgray', linestyle='--', lw=0.5)

        for y in y_grid:
            ax.axhline(y, color='lightgray', linestyle='--', lw=0.5)

        # Plotting other spots as background
        bg_index = [i for i, _ in enumerate(labels) if
                    i not in id_motif_celltype]  # the other spots are colored as background
        bg_adata = self.adata[bg_index, :]
        ax.scatter(bg_adata.obsm[self.spatial_key][:, 0],
                   bg_adata.obsm[self.spatial_key][:, 1],
                   color='darkgrey', s=1)

        for ct in motif:
            ct_ind = motif_spot_pos.obs[self.label_key] == ct
            ax.scatter(motif_spot_pos.obsm[self.spatial_key][ct_ind, 0],
                       motif_spot_pos.obsm[self.spatial_key][ct_ind, 1],
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
        plt.show()

    def plot_motif_rand(self,
                        motif: Union[str, List[str]],
                        max_dist: float = 100,
                        n_points: int = 1000,
                        min_count: int = 0,
                        min_support: float = 0.5,
                        min_size: int = 0,
                        if_max: bool = True,
                        fig_size: tuple = (10, 5),
                        seed: int = 2023,
                        ):
        """
        Display the random sampled points with motif in radius-based neighborhood,
        and cell types of motif in the neighborhood of these grid points. To make sure the input
        motif can be found in the results obtained by find_patterns_grid, use the same arguments
        as those in find_pattern_grid method.

        Parameter
        ---------
        motif:
            Motif (names of cell types) to be colored
        max_dist:
            Spacing distance for building grid. Make sure using the same value as that in find_patterns_grid.
        n_points:
            Number of random points to generate.
        min_count:
            Minimum number of each cell type to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        min_size:
            Minimum neighborhood size for each point to consider.
        if_max:
            Return all frequent patterns (if_max=False) or frequent patterns with maximal combinations (if_max=True).
            If a pattern (A, B, C) is frequent, its subsets (A, B), (A, C), (B, C) and (A), (B), (C) are also frequent.
            Return (A, B, C) if if_max=True otherwise return (A, B, C) and all its subsets.
        fig_size:
            Figure size.
        seed:
            Set random seed for reproducible.
        """
        if isinstance(motif, str):
            motif = [motif]

        cell_pos = self.adata.obsm[self.spatial_key]
        labels = self.adata.obs[self.label_key]
        labels_unique = labels.unique()
        motif_exc = [m for m in motif if m not in labels_unique]
        if len(motif_exc) != 0:
            print(f"Found no {motif_exc} in {self.label_key}. Ignoring them.")
        motif = [m for m in motif if m not in motif_exc]

        # Random sample points
        xmax, ymax = np.max(cell_pos, axis=0)
        xmin, ymin = np.min(cell_pos, axis=0)
        np.random.seed(seed)
        pos = np.column_stack((np.random.rand(n_points) * (xmax - xmin) + xmin,
                               np.random.rand(n_points) * (ymax - ymin) + ymin))

        # Compute fp here just to make sure we can use the same color map as in find_patterns_grid function.
        # If there's no need to keep same color map, can just use self.kd_tree.query() in knn or
        # self.kd_tree.query_ball_point in radisu-based neighborhood.
        fp, trans_df, _ = self.build_fptree_dist(cell_pos=pos,
                                                 max_dist=max_dist,
                                                 min_size=min_size,
                                                 min_count=min_count,
                                                 min_support=min_support)

        idxs = self.kd_tree.query_ball_point(pos, r=max_dist, return_sorted=True)

        if if_max:
            fp = self.find_maximal_patterns(fp=fp)

        # Locate the index of grid points acting as centers with motif nearby
        id_center = []
        for i, idx in enumerate(idxs):
            ns = [labels[id] for id in idx[1:]]
            if self.has_motif(neighbors=motif, labels=ns):
                id_center.append(i)

        # Locate the index of cell types contained in motif in the
        # neighborhood of above random points with motif nearby
        id_motif_celltype = set()  # the index of spots with cell types in motif and within the neighborhood of
        # above random sampled points
        for id in id_center:
            id_neighbor = [i for i in idxs[id][1:] if labels[i] in motif]
            id_motif_celltype.update(id_neighbor)

        # Plot above spots and center grid points
        # Set color map as in find_patterns_grid
        fp_cts = sorted(set(t for items in fp['itemsets'] for t in list(items)))
        n_colors = len(fp_cts)
        colors = sns.color_palette('hsv', n_colors)
        color_map = {ct: col for ct, col in zip(fp_cts, colors)}

        motif_spot_pos = self.adata[list(id_motif_celltype), :]
        fig, ax = plt.subplots(figsize=fig_size)
        ax.scatter(pos[id_center, 0], pos[id_center, 1], label='Random Sampling Points',
                   edgecolors='red', facecolors='none', s=8)

        # Plotting other spots as background
        bg_index = [i for i, _ in enumerate(labels) if
                    i not in id_motif_celltype]  # the other spots are colored as background
        bg_adata = self.adata[bg_index, :]
        ax.scatter(bg_adata.obsm[self.spatial_key][:, 0],
                   bg_adata.obsm[self.spatial_key][:, 1],
                   color='darkgrey', s=1)

        for ct in motif:
            ct_ind = motif_spot_pos.obs[self.label_key] == ct
            ax.scatter(motif_spot_pos.obsm[self.spatial_key][ct_ind, 0],
                       motif_spot_pos.obsm[self.spatial_key][ct_ind, 1],
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
        plt.show()