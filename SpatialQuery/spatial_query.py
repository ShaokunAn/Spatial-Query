from collections import Counter
from itertools import combinations
from typing import List, Union
import time

import matplotlib.pyplot as plt
import statsmodels.stats.multitest as mt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from pandas import DataFrame
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree
from scipy.stats import hypergeom
import spatial_module_utils
from sklearn.preprocessing import LabelEncoder


class spatial_query:
    def __init__(self,
                 adata: AnnData,
                 dataset: str = 'ST',
                 spatial_key: str = 'X_spatial',
                 label_key: str = 'predicted_label',
                 leaf_size: int = 10):
        if spatial_key not in adata.obsm.keys() or label_key not in adata.obs.keys():
            raise ValueError(f"The Anndata object must contain {spatial_key} in obsm and {label_key} in obs.")
        # Store spatial position and cell type label
        self.spatial_key = spatial_key
        self.spatial_pos = adata.obsm[self.spatial_key]
        self.dataset = dataset
        self.label_key = label_key
        self.labels = adata.obs[self.label_key]
        self.labels = self.labels.astype('category')
        self.kd_tree = KDTree(self.spatial_pos, leafsize=leaf_size)

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
        trans_suf = [f"{item}_{i}" for item, value in counter.items() for i in range(value)]
        # trans_suf = [f"{item}_{value}" for item, value in counter.items()]
        # count_dict = defaultdict(int)
        # for i, item in enumerate(transaction):
        #     # Increment the count for the item, or initialize it if it's new
        #     count_dict[item] += 1
        #     # Update the item with its count as suffix
        #     transaction[i] = f"{item}_{count_dict[item]}"
        # return transaction
        return trans_suf

    @staticmethod
    def _remove_suffix(fp: pd.DataFrame):
        """
        Remove the suffix of frequent patterns.
        """
        trans = [list(tran) for tran in fp['itemsets'].values]
        fp_no_suffix = [[item.split('_')[0] for item in tran] for tran in trans]
        # Create a DataFrame
        fp['itemsets'] = fp_no_suffix
        return fp

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
        # maximal_patterns_ = [list(p) for p in maximal_patterns]

        # Filter the original DataFrame to keep only the maximal patterns
        return fp[fp['itemsets'].isin(maximal_patterns)].reset_index(drop=True)

    def find_fp_knn(self,
                    ct: str,
                    k: int = 30,
                    min_support: float = 0.5,
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

        Return
        ------
        Frequent patterns in the neighborhood of certain cell type.
        """
        if ct not in self.labels.unique():
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        cinds = [id for id, l in enumerate(self.labels) if l == ct]
        ct_pos = self.spatial_pos[cinds]

        fp, _, _ = self.build_fptree_knn(cell_pos=ct_pos, k=k,
                                         min_support=min_support,
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

        fp, _, _ = self.build_fptree_dist(cell_pos=ct_pos,
                                          max_dist=max_dist,
                                          min_size=min_size,
                                          min_support=min_support,
                                          cinds=cinds)

        return fp

    def motif_enrichment_knn(self,
                             ct: str,
                             motifs: Union[str, List[str]] = None,
                             k: int = 30,
                             min_support: float = 0.5,
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
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        max_dist:
            Maximum distance for neighbors (default: 200).

        Return
        ------
        pd.Dataframe containing the cell type name, motifs, number of motifs nearby given cell type,
        number of spots of cell type, number of motifs in single FOV, p value of hypergeometric distribution.
        """
        if ct not in self.labels.unique():
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        dists, idxs = self.kd_tree.query(self.spatial_pos,
                                         k=k + 1)  # use k+1 to find the knn except for the points themselves
        dists = np.array(dists)
        idxs = np.array(idxs)  # c++ can access Numpy directly without duplicating data

        cinds = [i for i, l in enumerate(self.labels) if l == ct]

        out = []
        if motifs is None:
            fp = self.find_fp_knn(ct=ct, k=k,
                                  min_support=min_support,
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

        for motif in motifs:
            motif = list(motif) if not isinstance(motif, list) else motif
            sort_motif = sorted(motif)

            label_encoder = LabelEncoder()
            int_labels = label_encoder.fit_transform(self.labels)
            int_ct = label_encoder.transform(np.array(ct, dtype=object, ndmin=1))
            int_motifs = label_encoder.transform(np.array(motif))

            dists, idxs = self.kd_tree.query(self.spatial_pos, k=k+1)
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

            # 过滤包含特定细胞类型的cells
            mask = int_labels == int_ct
            n_motif_ct = np.sum(np.all(neighbor_counts[mask][:, int_motifs] > 0, axis=1))
            n_motif_labels = np.sum(np.all(neighbor_counts[:, int_motifs] > 0, axis=1))

            # n_motif_ct, n_motif_labels = spatial_module_utils.search_motif_knn(
            #     motif, idxs, dists, self.labels, cinds, max_dist,
            # )

            n_ct = len(cinds)
            if ct in motif:
                n_ct = round(n_ct / motif.count(ct))

            hyge = hypergeom(M=len(self.labels), n=n_ct, N=n_motif_labels)
            # M is number of total, N is number of drawn without replacement, n is number of success in total
            motif_out = {'center': ct, 'motifs': sort_motif, 'n_center_motif': n_motif_ct,
                         'n_center': n_ct, 'n_motif': n_motif_labels, 'p-values': hyge.sf(n_motif_ct)}
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
                              max_dist: float = 100,
                              min_size: int = 0,
                              min_support: float = 0.5,
                              max_ns: int = 1000000) -> DataFrame:
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
        Returns
        -------
        Tuple containing counts and statistical measures.
        """
        if ct not in self.labels.unique():
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        idxs = self.kd_tree.query_ball_point(self.spatial_pos, r=max_dist, return_sorted=True)
        cinds = [i for i, label in enumerate(self.labels) if label == ct]
        idxs = idxs.tolist()

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

        for motif in motifs:
            motif = list(motif) if not isinstance(motif, list) else motif
            sort_motif = sorted(motif)
            n_motif_ct, n_motif_labels = spatial_module_utils.search_motif_dist(
                motif, idxs, self.labels, cinds, max_ns

            )

            # n_motif_ct = 0
            # for i in cinds:
            #     e = min(len(idxs[i]), max_ns)
            #     if self.has_motif(sort_motif, [self.labels[idx] for idx in idxs[i][:e] if idx != i]):
            #         n_motif_ct += 1
            #
            # n_motif_labels = 0
            # for i in range(len(idxs)):
            #     e = min(len(idxs[i]), max_ns)
            #     if self.has_motif(sort_motif, [self.labels[idx] for idx in idxs[i][:e] if idx != i]):
            #         n_motif_labels += 1

            n_ct = len(cinds)
            if ct in motif:
                n_ct = round(n_ct / motif.count(ct))

            hyge = hypergeom(M=len(self.labels), n=n_ct, N=n_motif_labels)
            motif_out = {'center': ct, 'motifs': sort_motif, 'n_center_motif': n_motif_ct,
                         'n_center': n_ct, 'n_motif': n_motif_labels, 'p-values': hyge.sf(n_motif_ct)}
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

    def build_fptree_dist(self,
                          cell_pos: np.ndarray = None,
                          max_dist: float = 100,
                          min_support: float = 0.5,
                          if_max: bool = True,
                          min_size: int = 0,
                          cinds: List[int] = None,
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
        if_max:
            By default return the maximum set of frequent patterns without the subsets. If if_max=False, return all
            patterns whose support values are greater than min_support.
        min_size:
            Minimum neighborhood size for each point to consider.
        max_ns:
            Maximum number of neighborhood size for each point.

        Return
        ------
        A tuple containing the FPs, the transactions table and the nerghbors index.
        """
        if cell_pos is None:
            cell_pos = self.spatial_pos

        start = time.time()
        idxs = self.kd_tree.query_ball_point(cell_pos, r=max_dist, return_sorted=False)
        if cinds is None:
            cinds = list(range(len(idxs)))
        end = time.time()
        print("query: {end-start} seconds")

        # Prepare data for FP-Tree construction
        start = time.time()
        transactions = []
        valid_idxs = []
        labels = np.array(self.labels)
        for i_idx, idx in zip(cinds, idxs):
            idx_array = np.array(idx)
            valid_mask = idx_array != i_idx
            valid_indices = idx_array[valid_mask][:max_ns]

            transaction = labels[valid_indices]
            if len(transaction) > min_size:
                transactions.append(transaction.tolist())
                valid_idxs.append(idx)
        end = time.time()
        print(f"build transactions: {end-start} seconds")
        # Convert transactions to a DataFrame suitable for fpgrowth
        start = time.time()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        # Construct FP-Tree using fpgrowth
        fp_tree = fpgrowth(df, min_support=min_support, use_colnames=True)
        end = time.time()
        print(f"fp_growth: {end-start} seconds")
        if if_max:
            start = time.time()
            fp_tree = self.find_maximal_patterns(fp=fp_tree)
            end = time.time()
            print(f"find_maximal_patterns: {end-start} seconds")

        # Remove suffix of items if treating duplicates as different items
        # if dis_duplicates:
        #     fp_tree = self._remove_suffix(fp_tree)

        if len(fp_tree) == 0:
            return pd.DataFrame(columns=['support', 'itemsets']), df, valid_idxs
        else:
            fp_tree['itemsets'] = fp_tree['itemsets'].apply(lambda x: tuple(sorted(x)))
            fp_tree = fp_tree.drop_duplicates().reset_index(drop=True)
            fp_tree['itemsets'] = fp_tree['itemsets'].apply(lambda x: list(x))
            fp_tree = fp_tree.sort_values(by='support', ignore_index=True, ascending=False)
            return fp_tree, df, valid_idxs

    def build_fptree_knn(self,
                         cell_pos: np.ndarray = None,
                         k: int = 30,
                         min_support: float = 0.5,
                         max_dist: float = 500,
                         if_max: bool = True
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
        max_dist:
            The maximum distance at which points are considered neighbors.
        if_max:
            By default return the maximum set of frequent patterns without the subsets. If if_max=False, return all
            patterns whose support values are greater than min_support.

        Return
        ------
        A tuple containing the FPs, the transactions table, and the neighbors index.
        """
        if cell_pos is None:
            cell_pos = self.spatial_pos
        start = time.time()
        dists, idxs = self.kd_tree.query(cell_pos, k=k + 1)
        end = time.time()
        print(f"knn query: {end-start} seconds")

        # Prepare data for FP-Tree construction
        start = time.time()
        idxs = np.array(idxs)
        dists = np.array(dists)
        labels = np.array(self.labels)
        transactions = []
        mask = dists < max_dist
        for i, idx in enumerate(idxs):
            inds = idx[mask[i]]
            transaction = labels[inds[1:]]
            # if dis_duplicates:
            #     transaction = distinguish_duplicates_numpy(transaction)
            transactions.append(transaction)  # 将 NumPy 数组转换回列表

        end = time.time()
        print(f"build transactions: {end-start} seconds")

        # transactions = []
        # for i, idx in enumerate(idxs):
        #     inds = [id for j, id in enumerate(idx) if
        #             dists[i][j] < max_dist]  # only contain the KNN whose distance is less than max_dist
        #     transaction = [self.labels[i] for i in inds[1:] if self.labels[i]]
        #     # if dis_duplicates:
        #     #     transaction = self._distinguish_duplicates(transaction)
        #     transactions.append(transaction)

        # Convert transactions to a DataFrame suitable for fpgrowth
        start = time.time()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        # Construct FP-Tree using fpgrowth
        fp_tree = fpgrowth(df, min_support=min_support, use_colnames=True)
        end = time.time()
        print(f"fp-growth: {end-start} seconds")

        if if_max:
            start = time.time()
            fp_tree = self.find_maximal_patterns(fp_tree)
            end = time.time()
            print(f"find_maximal_patterns: {end-start} seconds")

        # if dis_duplicates:
        #     fp_tree = self._remove_suffix(fp_tree)
        if len(fp_tree) == 0:
            return pd.DataFrame(columns=['support', 'itemsets']), df, idxs
        else:
            start = time.time()
            fp_tree['itemsets'] = fp_tree['itemsets'].apply(lambda x: tuple(sorted(x)))
            fp_tree = fp_tree.drop_duplicates().reset_index(drop=True)
            fp_tree['itemsets'] = fp_tree['itemsets'].apply(lambda x: list(x))
            fp_tree = fp_tree.sort_values(by='support', ignore_index=True, ascending=False)
            end = time.time()
            print(f"format output: {end-start} seconds")
            return fp_tree, df, idxs

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
        xmax, ymax = np.max(self.spatial_pos, axis=0)
        xmin, ymin = np.min(self.spatial_pos, axis=0)
        x_grid = np.arange(xmin - max_dist, xmax + max_dist, max_dist)
        y_grid = np.arange(ymin - max_dist, ymax + max_dist, max_dist)
        grid = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)

        fp, trans_df, idxs = self.build_fptree_dist(cell_pos=grid,
                                                    max_dist=max_dist, min_size=min_size,
                                                    min_support=min_support)

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
            fp['cell_id'] = id_neighbor_motifs

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
        xmax, ymax = np.max(self.spatial_pos, axis=0)
        xmin, ymin = np.min(self.spatial_pos, axis=0)
        np.random.seed(seed)
        pos = np.column_stack((np.random.rand(n_points) * (xmax - xmin) + xmin,
                               np.random.rand(n_points) * (ymax - ymin) + ymin))

        fp, trans_df, idxs = self.build_fptree_dist(cell_pos=pos,
                                                    max_dist=max_dist, min_size=min_size,
                                                    min_support=min_support,
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
            fp['cell_id'] = id_neighbor_motifs

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

        plt.show()

    def plot_motif_grid(self,
                        motif: Union[str, List[str]],
                        fp: pd.DataFrame,
                        fig_size: tuple = (10, 5),
                        max_dist: float = 100,
                        ):
        """
        Display the distribution of each motif around grid points. To make sure the input
        motif can be found in the results obtained by find_patterns_grid, use the same arguments
        as those in find_pattern_grid method.

        Parameter
        ---------
        motif:
            Motif (names of cell types) to be colored
        fp:
            Frequent patterns identified by find_patterns_grid.
        max_dist:
            Spacing distance for building grid. Make sure using the same value as that in find_patterns_grid.
        fig_size:
            Figure size.
        """
        if isinstance(motif, str):
            motif = [motif]

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
        idxs = self.kd_tree.query_ball_point(grid, r=max_dist, return_sorted=True)

        # Locate the index of grid points acting as centers with motif nearby
        id_center = []
        for i, idx in enumerate(idxs):
            ns = [self.labels[id] for id in idx]
            if self.has_motif(neighbors=motif, labels=ns):
                id_center.append(i)

        # Locate the index of cell types contained in motif in the
        # neighborhood of above grid points with motif nearby
        id_motif_celltype = fp[fp['itemsets'].apply(
            lambda p: set(p)) == set(motif)]
        id_motif_celltype = id_motif_celltype['cell_id'].iloc[0]

        # Plot above spots and center grid points
        # Set color map as in find_patterns_grid
        fp_cts = sorted(set(t for items in fp['itemsets'] for t in list(items)))
        n_colors = len(fp_cts)
        colors = sns.color_palette('hsv', n_colors)
        color_map = {ct: col for ct, col in zip(fp_cts, colors)}

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

        motif_unique = list(set(motif))
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
        plt.show()

    def plot_motif_rand(self,
                        motif: Union[str, List[str]],
                        fp: pd.DataFrame,
                        max_dist: float = 100,
                        n_points: int = 1000,
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
        fp:
            Frequent patterns identified by find_patterns_grid.
        max_dist:
            Spacing distance for building grid. Make sure using the same value as that in find_patterns_grid.
        n_points:
            Number of random points to generate.
        fig_size:
            Figure size.
        seed:
            Set random seed for reproducible.
        """
        if isinstance(motif, str):
            motif = [motif]

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

        idxs = self.kd_tree.query_ball_point(pos, r=max_dist, return_sorted=True)

        # Locate the index of grid points acting as centers with motif nearby
        id_center = []
        for i, idx in enumerate(idxs):
            ns = [self.labels[id] for id in idx]
            if self.has_motif(neighbors=motif, labels=ns):
                id_center.append(i)

        # Locate the index of cell types contained in motif in the
        # neighborhood of above random points with motif nearby
        id_motif_celltype = fp[fp['itemsets'].apply(
            lambda p: set(p)) == set(motif)]
        id_motif_celltype = id_motif_celltype['cell_id'].iloc[0]

        # Plot above spots and center grid points
        # Set color map as in find_patterns_grid
        fp_cts = sorted(set(t for items in fp['itemsets'] for t in list(items)))
        n_colors = len(fp_cts)
        colors = sns.color_palette('hsv', n_colors)
        color_map = {ct: col for ct, col in zip(fp_cts, colors)}

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
        motif_unique = list(set(motif))
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
        plt.show()

    def plot_motif_celltype(self,
                            ct: str,
                            motif: Union[str, List[str]],
                            max_dist: float = 100,
                            fig_size: tuple = (10, 5)
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
        """
        if isinstance(motif, str):
            motif = [motif]

        motif_exc = [m for m in motif if m not in self.labels.unique()]
        if len(motif_exc) != 0:
            print(f"Found no {motif_exc} in {self.label_key}. Ignoring them.")
        motif = [m for m in motif if m not in motif_exc]

        if ct not in self.labels.unique():
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        cinds = [i for i, label in enumerate(self.labels) if label == ct]  # id of center cell type
        # ct_pos = self.spatial_pos[cinds]
        idxs = self.kd_tree.query_ball_point(self.spatial_pos, r=max_dist, return_sorted=True)

        # find the index of cell type spots whose neighborhoods contain given motif
        cind_with_motif = []
        sort_motif = sorted(motif)
        for id in cinds:
            if self.has_motif(sort_motif, [self.labels[idx] for idx in idxs[id] if idx != id]):
                cind_with_motif.append(id)

        # Locate the index of motifs in the neighborhood of center cell type.
        id_motif_celltype = set()
        for id in cind_with_motif:
            id_neighbor = [i for i in idxs[id] if self.labels[i] in motif and i != id]
            id_motif_celltype.update(id_neighbor)

        # Plot figures
        motif_unique = set(motif)
        n_colors = len(motif_unique)
        colors = sns.color_palette('hsv', n_colors)
        color_map = {ct: col for ct, col in zip(motif_unique, colors)}
        motif_spot_pos = self.spatial_pos[list(id_motif_celltype), :]
        motif_spot_label = self.labels[list(id_motif_celltype)]
        fig, ax = plt.subplots(figsize=fig_size)
        # Plotting other spots as background
        bg_index = [i for i, _ in enumerate(self.labels) if i not in list(id_motif_celltype) + cind_with_motif]
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
        plt.show()
