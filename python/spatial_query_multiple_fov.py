import numpy as np
from pandas import DataFrame

from spatial_query import spatial_query
from anndata import AnnData
from typing import List, Union
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from scipy.stats import hypergeom


class spatial_query_multi:
    def __init__(self,
                 adatas: List[AnnData],
                 datasets: List[str],
                 spatial_key: str,
                 label_key: str,
                 leaf_size: int):
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
        """
        # Each element in self.spatial_queries stores a spatial_query object
        self.spatial_key = spatial_key
        self.label_key = label_key
        self.datasets = datasets
        self.spatial_queries = [spatial_query(adata=adata, dataset=self.datasets[i],
                                              spatial_key=spatial_key,
                                              label_key=label_key,
                                              leaf_size=leaf_size) for i, adata in enumerate(adatas)]

    def find_fp_knn(self,
                    ct: str,
                    dataset: Union[str, List[str]] = None,
                    k: int = 20,
                    min_count: int = 0,
                    min_support: float = 0.5,
                    if_max: bool = True,
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
        # Search transactions for each field of view, find the frequent patterns of integrated transactions
        if_exist_label = [ct in s.adata.obs[self.label_key].unique() for s in self.spatial_queries]
        if not any(if_exist_label):
            raise ValueError(f"Not found {self.label_key} in all datasets!")

        if dataset is None:
            dataset = [s.dataset for s in self.spatial_queries]

        if isinstance(dataset, str):
            dataset = [dataset]

        transactions = []
        for s in self.spatial_queries:
            if s.dataset not in dataset:
                continue
            cell_pos = s.adata.obsm[self.spatial_key]
            labels = s.adata.obs[self.label_key]
            if ct not in labels.unique():
                continue

            cinds = [id for id, l in enumerate(labels) if l == ct]
            ct_pos = cell_pos[cinds]

            dists, idxs = s.kd_tree.query(ct_pos, k=k + 1)
            ct_all = sorted(set(labels))
            ct_count = np.zeros(len(ct_all), dtype=int)
            for i, idx in enumerate(idxs):
                for j in idx[1:len(idx)]:
                    ct_count[ct_all.index(labels[j])] += 1
            ct_exclude = [ct_all[i] for i, count in enumerate(ct_count) if count < min_count]

            for idx in idxs:
                transaction = [labels[i] for i in idx[1:len(idx)] if labels[i] not in ct_exclude]
                transactions.append(transaction)

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        fp = fpgrowth(df, min_support=min_support, use_colnames=True)

        if if_max:
            fp = spatial_query.find_maximal_patterns(fp=fp)

        return fp

    def find_fp_dist(self,
                     ct: str,
                     dataset: Union[str, List[str]] = None,
                     max_dist: float = 100,
                     min_size: int = 0,
                     min_count: int = 0,
                     min_support: float = 0.5,
                     if_max: bool = True,
                     max_ns: int = 1000000
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

        Return
        ------
        Frequent patterns in the neighborhood of certain cell type.
        """
        # Search transactions for each field of view, find the frequent patterns of integrated transactions
        if_exist_label = [ct in s.adata.obs[self.label_key].unique() for s in self.spatial_queries]
        if not any(if_exist_label):
            raise ValueError(f"Not found {self.label_key} in any datasets!")

        if dataset is None:
            dataset = [s.dataset for s in self.spatial_queries]
        if isinstance(dataset, str):
            dataset = [dataset]

        transactions = []
        for s in self.spatial_queries:
            if s.dataset not in dataset:
                continue
            cell_pos = s.adata.obsm[self.spatial_key]
            labels = s.adata.obs[self.label_key]
            if ct not in labels.unique():
                continue

            cinds = [id for id, l in enumerate(labels) if l == ct]
            ct_pos = cell_pos[cinds]

            pos2neighbor_count = {}
            idxs = s.kd_tree.query_ball_point(ct_pos, r=max_dist, return_sorted=True)
            ct_all = sorted(set(labels))
            ct_count = np.zeros(len(ct_all), dtype=int)

            for i, idx in enumerate(idxs):
                if len(idx) > min_size + 1:
                    for j in idx[1:min(max_ns, len(idx))]:
                        ct_count[ct_all.index(labels[j])] += 1
                pos2neighbor_count[tuple(cell_pos[i])] = len(idx)
            ct_exclude = [ct_all[i] for i, count in enumerate(ct_count) if count < min_count]

            for idx in idxs:
                transaction = [labels[i] for i in idx[1:min(max_ns, len(idx))] if labels[i] not in ct_exclude]
                if len(transaction) > min_size:
                    transactions.append(transaction)

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        fp = fpgrowth(df, min_support=min_support, use_colnames=True)

        if if_max:
            fp = spatial_query.find_maximal_patterns(fp=fp)

        return fp

    def motif_enrichment_knn(self,
                             ct: str,
                             motifs: Union[str, List[str]] = None,
                             dataset: Union[str, List[str]] = None,
                             k: int = 20,
                             min_count: int = 0,
                             min_support: float = 0.5,
                             if_max: bool = True,
                             max_dist=float('inf')
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
        min_count:
            Minimum number of each cell type to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.
        if_max:
            Return all frequent patterns (if_max=False) or frequent patterns with maximal combinations (if_max=True).
            If a pattern (A, B, C) is frequent, its subsets (A, B), (A, C), (B, C) and (A), (B), (C) are also frequent.
            Return (A, B, C) if if_max=True otherwise return (A, B, C) and all its subsets.
        max_dist:
            Maximum distance for neighbors (default: infinity).

        Return
        ------
        pd.Dataframe containing the cell type name, motifs, number of motifs nearby given cell type,
        number of spots of cell type, number of motifs in single FOV, p value of hypergeometric distribution.
        """
        if dataset is None:
            dataset = [s.dataset for s in self.spatial_queries]
        if isinstance(dataset, str):
            dataset = [dataset]

        out = []
        if_exist_label = [ct in s.adata.obs[self.label_key].unique() for s in self.spatial_queries]
        if not any(if_exist_label):
            raise ValueError(f"Not found {self.label_key} in any datasets!")

        # Check whether specify motifs. If not, search frequent patterns among specified datasets
        # and use them as interested motifs
        if motifs is None:
            fp = self.find_fp_knn(ct=ct, k=k, dataset=dataset, min_count=min_count,
                                  min_support=min_support, if_max=if_max)
            motifs = fp['itemsets']
        else:
            if isinstance(motifs, str):
                motifs = [motifs]

            labels_unique_all = set([s.adata.obs[self.label_key].unique() for s in self.spatial_queries])
            motifs_exc = [m for m in motifs if m not in labels_unique_all]
            if len(motifs_exc) != 0:
                print(f"Not found {motifs_exc} in {dataset}. Ignoring them.")
            motifs = [m for m in motifs if m not in motifs_exc]
            motifs = [motifs]

        for motif in motifs:
            n_labels = 0
            n_ct = 0
            n_motif_labels = 0
            n_motif_ct = 0

            motif = list(motif) if not isinstance(motif, list) else motif
            sort_motif = sorted(motif)

            # Calculate statistics of each dataset
            for s in self.spatial_queries:
                if s.dataset not in dataset:
                    continue
                cell_pos = s.adata.obsm[self.spatial_key]
                labels = s.adata.obs[self.label_key]
                if ct not in labels.unique():
                    continue
                dists, idxs = s.kd_tree.query(cell_pos, k=k+1)
                cinds = [i for i, l in enumerate(labels) if l == ct]

                for i in cinds:
                    inds = [ind for ind, d in enumerate(dists[i]) if d < max_dist]
                    if len(inds) > 1:
                        if spatial_query.has_motif(sort_motif, [labels[idx] for idx in idxs[i][inds[1:]]]):
                            n_motif_ct += 1

                for i in range(len(labels)):
                    if spatial_query.has_motif(sort_motif, [labels[idx] for idx in idxs[i][1:]]):
                        n_motif_labels += 1

                n_ct += len(cinds)
                n_labels += len(labels)

            if ct in motif:
                n_ct = round(n_ct/motif.count(ct))

            hyge = hypergeom(M=n_labels, n=n_ct, N=n_motif_labels)
            motif_out = {'center': ct, 'motifs': sort_motif, 'n_center_motif': n_motif_ct,
                         'n_center': n_ct, 'n_motif': n_motif_labels, 'p-val': hyge.sf(n_motif_ct)}
            out.append(motif_out)

        out_pd = pd.DataFrame(out)
        out_pd = out_pd.sort_values(by='p-val', ignore_index=True)

        # TODO: add a multiple testing correction procedure here!

        return out_pd

    def motif_enrichment_dist(self,
                              ct: str,
                              motifs: Union[str, List[str]] = None,
                              dataset: Union[str, List[str]] = None,
                              max_dist: float = 100,
                              min_size: int = 0,
                              min_count: int = 0,
                              min_support: float = 0.5,
                              if_max: bool = True,
                              max_ns: int = 1000000) -> DataFrame:
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
        if dataset is None:
            dataset = [s.dataset for s in self.spatial_queries]
        if isinstance(dataset, str):
            dataset = [dataset]

        out = []
        if_exist_label = [ct in s.adata.obs[self.label_key].unique() for s in self.spatial_queries]
        if not any(if_exist_label):
            raise ValueError(f"Not found {self.label_key} in any datasets!")

        # Check whether specify motifs. If not, search frequent patterns among specified datasets
        # and use them as interested motifs
        if motifs is None:
            fp = self.find_fp_dist(ct=ct, dataset=dataset, max_dist=max_dist, min_size=min_size,
                                   min_count=min_count, min_support=min_support, if_max=if_max)
            motifs = fp['itemsets']
        else:
            if isinstance(motifs, str):
                motifs = [motifs]

            labels_unique_all = set([s.adata.obs[self.label_key].unique() for s in self.spatial_queries])
            motifs_exc = [m for m in motifs if m not in labels_unique_all]
            if len(motifs_exc) != 0:
                print(f"Not found {motifs_exc} in {dataset}! Ignoring them.")
            motifs = [m for m in motifs if m not in motifs_exc]
            motifs = [motifs]

        for motif in motifs:
            n_labels = 0
            n_ct = 0
            n_motif_labels = 0
            n_motif_ct = 0

            motif = list(motif) if not isinstance(motif, list) else motif
            sort_motif = sorted(motif)

            for s in self.spatial_queries:
                if s.dataset not in dataset:
                    continue
                cell_pos = s.adata.obsm[self.spatial_key]
                labels = s.adata.obs[self.label_key]

                if ct not in labels.unique():
                    continue

                idxs = s.kd_tree.query_ball_point(cell_pos, r=max_dist, return_sorted=True)
                cinds = [i for i, label in enumerate(labels) if label == ct]

                for i in cinds:
                    e = min(len(idxs[i]), max_ns)
                    if spatial_query.has_motif(sort_motif, [labels[idx] for idx in idxs[i][1:e]]):
                        n_motif_ct += 1

                for i in range(len(idxs)):
                    e = min(len(idxs[i]), max_ns)
                    if spatial_query.has_motif(sort_motif, [labels[idx] for idx in idxs[i][1:e]]):
                        n_motif_labels += 1

                n_ct += len(cinds)
                n_labels += len(labels)

            if ct in motif:
                n_ct = round(n_ct / motif.count(ct))

            hyge = hypergeom(M=n_labels, n=n_ct, N=n_motif_labels)
            motif_out = {'center': ct, 'motifs': sort_motif, 'n_center_motif': n_motif_ct,
                         'n_center': n_ct, 'n_motif': n_motif_labels, 'p-val': hyge.sf(n_motif_ct)}
            out.append(motif_out)

        out_pd = pd.DataFrame(out)
        out_pd = out_pd.sort_values(by='p-val', ignore_index=True)
        # TODO: add multiple testing correction procedure here!
        return out_pd
