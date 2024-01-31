from typing import List, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multitest as mt
from anndata import AnnData
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from pandas import DataFrame
from scipy.stats import hypergeom

from .spatial_query import spatial_query


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
                    dis_duplicates: bool = False,
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
        dis_duplicates:
            Distinguish duplicates in patterns if dis_duplicates=True. This will consider transactions within duplicates
            like (A, A, A, B, C) otherwise only patterns with unique cell types will be considered like (A, B, C).

        Return
        ------
        Frequent patterns in the neighborhood of certain cell type.
        """
        # Search transactions for each field of view, find the frequent patterns of integrated transactions
        if_exist_label = [ct in s.labels.unique() for s in self.spatial_queries]
        if not any(if_exist_label):
            raise ValueError(f"Found no {self.label_key} in all datasets!")

        if dataset is None:
            # Use all datasets if dataset is not provided
            dataset = [s.dataset.split('_')[0] for s in self.spatial_queries]

        # Make sure dataset is a list
        if isinstance(dataset, str):
            dataset = [dataset]

        transactions = []
        for s in self.spatial_queries:
            if s.dataset.split('_')[0] not in dataset:
                continue
            cell_pos = s.spatial_pos
            labels = s.labels
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
                transaction = [labels[i] for i in idx[1:] if labels[i] not in ct_exclude]
                if dis_duplicates:
                    transaction = s._distinguish_duplicates(transaction)
                transactions.append(transaction)

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        fp = fpgrowth(df, min_support=min_support, use_colnames=True)

        fp = spatial_query.find_maximal_patterns(fp=fp)

        # Remove suffix of items if treating duplicates as different items
        if dis_duplicates:
            fp = spatial_query._remove_suffix(fp)

        return fp

    def find_fp_dist(self,
                     ct: str,
                     dataset: Union[str, List[str]] = None,
                     max_dist: float = 100,
                     min_size: int = 0,
                     min_count: int = 0,
                     min_support: float = 0.5,
                     dis_duplicates: bool = False,
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
        dis_duplicates:
            Distinguish duplicates in patterns if dis_duplicates=True. This will consider transactions within duplicates
            like (A, A, A, B, C) otherwise only patterns with unique cell types will be considered like (A, B, C).
        max_ns:
            Maximum number of neighborhood size for each point.

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

        transactions = []
        for s in self.spatial_queries:
            if s.dataset.split('_')[0] not in dataset:
                continue
            cell_pos = s.spatial_pos
            labels = s.labels
            if ct not in labels.unique():
                continue

            cinds = [id for id, l in enumerate(labels) if l == ct]
            ct_pos = cell_pos[cinds]

            idxs = s.kd_tree.query_ball_point(ct_pos, r=max_dist, return_sorted=True)
            ct_all = sorted(set(labels))
            ct_count = np.zeros(len(ct_all), dtype=int)

            for i, idx in enumerate(idxs):
                if len(idx) > min_size + 1:
                    for j in idx[1:min(max_ns, len(idx))]:
                        ct_count[ct_all.index(labels[j])] += 1
            ct_exclude = [ct_all[i] for i, count in enumerate(ct_count) if count < min_count]

            for idx in idxs:
                transaction = [labels[i] for i in idx[1:min(max_ns, len(idx))] if labels[i] not in ct_exclude]
                if len(transaction) > min_size:
                    if dis_duplicates:
                        transaction = s._distinguish_duplicates(transaction)
                    transactions.append(transaction)

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        fp = fpgrowth(df, min_support=min_support, use_colnames=True)

        fp = spatial_query.find_maximal_patterns(fp=fp)

        # Remove suffix of items if treating duplicates as different items
        if dis_duplicates:
            fp = spatial_query._remove_suffix(fp)

        return fp

    def motif_enrichment_knn(self,
                             ct: str,
                             motifs: Union[str, List[str]] = None,
                             dataset: Union[str, List[str]] = None,
                             k: int = 20,
                             min_count: int = 0,
                             min_support: float = 0.5,
                             dis_duplicates: bool = False,
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
        dis_duplicates:
            Distinguish duplicates in patterns if dis_duplicates=True. This will consider transactions within duplicates
            like (A, A, A, B, C) otherwise only patterns with unique cell types will be considered like (A, B, C).
        max_dist:
            Maximum distance for neighbors (default: infinity).

        Return
        ------
        pd.Dataframe containing the cell type name, motifs, number of motifs nearby given cell type,
        number of spots of cell type, number of motifs in single FOV, p value of hypergeometric distribution.
        """
        if dataset is None:
            dataset = [s.dataset.split('_')[0] for s in self.spatial_queries]
        if isinstance(dataset, str):
            dataset = [dataset]

        out = []
        if_exist_label = [ct in s.labels.unique() for s in self.spatial_queries]
        if not any(if_exist_label):
            raise ValueError(f"Found no {self.label_key} in any datasets!")

        # Check whether specify motifs. If not, search frequent patterns among specified datasets
        # and use them as interested motifs
        if motifs is None:
            fp = self.find_fp_knn(ct=ct, k=k, dataset=dataset, min_count=min_count,
                                  min_support=min_support, dis_duplicates=dis_duplicates)
            motifs = fp['itemsets']
        else:
            if isinstance(motifs, str):
                motifs = [motifs]
            all_labels = pd.concat([s.labels for s in self.spatial_queries])
            labels_unique_all = set(all_labels.unique())
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
            for s in self.spatial_queries:
                if s.dataset.split('_')[0] not in dataset:
                    continue
                cell_pos = s.spatial_pos
                labels = s.labels
                if ct not in labels.unique():
                    continue
                dists, idxs = s.kd_tree.query(cell_pos, k=k + 1)
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
                n_ct = round(n_ct / motif.count(ct))

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
                              dis_duplicates: bool = False,
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

        # Check whether specify motifs. If not, search frequent patterns among specified datasets
        # and use them as interested motifs
        if motifs is None:
            fp = self.find_fp_dist(ct=ct, dataset=dataset, max_dist=max_dist, min_size=min_size,
                                   min_count=min_count, min_support=min_support, dis_duplicates=dis_duplicates)
            motifs = fp['itemsets']
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
                labels = s.labels

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

    def find_fp_knn_fov(self,
                        ct: str,
                        dataset_i: str,
                        k: int = 20,
                        min_count: int = 0,
                        min_support: float = 0.5,
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
        min_count:
            Minimum number of each cell type to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.

        Return
        ------
            Frequent patterns in the neighborhood of certain cell type.
        """
        if dataset_i not in self.datasets:
            raise ValueError(f"Found no {dataset_i.split('_')[0]} in any datasets.")

        sp_object = self.spatial_queries[self.datasets.index(dataset_i)]
        cell_pos = sp_object.spatial_pos
        labels = sp_object.labels
        if ct not in labels.unique():
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        cinds = [id for id, l in enumerate(labels) if l == ct]
        ct_pos = cell_pos[cinds]

        # Identify frequent patterns of cell types, including those subsets of patterns
        # whose support value exceeds min_support. Focus solely on the multiplicity
        # of cell types, rather than their frequency.
        fp, _, _ = sp_object.build_fptree_knn(cell_pos=ct_pos,
                                              k=k,
                                              min_count=min_count,
                                              min_support=min_support,
                                              dis_duplicates=False,
                                              if_max=False
                                              )
        return fp

    def find_fp_dist_fov(self,
                         ct: str,
                         dataset_i: str,
                         max_dist: float = 100,
                         min_size: int = 0,
                         min_count: int = 0,
                         min_support: float = 0.5,
                         ):
        """
        Find frequent patterns within the radius-based neighborhood of specific cell type of interest in single field of view.

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
        min_count:
            Minimum number of each cell type to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.

        Return
        ------
            Frequent patterns in the neighborhood of certain cell type.
        """
        if dataset_i not in self.datasets:
            raise ValueError(f"Found no {dataset_i.split('_')[0]} in any datasets.")

        sp_object = self.spatial_queries[self.datasets.index(dataset_i)]
        cell_pos = sp_object.spatial_pos
        labels = sp_object.labels
        if ct not in labels.unique():
            raise ValueError(f"Found no {ct} in {self.label_key}!")

        cinds = [id for id, l in enumerate(labels) if l == ct]
        ct_pos = cell_pos[cinds]

        fp, _, _ = sp_object.build_fptree_dist(cell_pos=ct_pos,
                                               max_dist=max_dist,
                                               min_support=min_support,
                                               min_count=min_count,
                                               min_size=min_size,
                                               dis_duplicates=False,
                                               if_max=False
                                               )
        return fp

    def differential_analysis_knn(self,
                                  ct: str,
                                  datasets: List[str],
                                  k: int = 20,
                                  min_count: int = 0,
                                  min_support: float = 0.5
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
        min_count:
            Minimum number of each cell type to consider.
        min_support:
            Threshold of frequency to consider a pattern as a frequent pattern.

        Return
        ------
            Dataframes with significant enriched patterns in differential analysis
        """
        if len(datasets) != 2:
            raise ValueError("Require 2 datasets for differential analysis.")
        # Identify frequent patterns for each dataset

        flag = 0
        # Identify frequent patterns in each dataset
        for d in datasets:
            fp_d = {}
            dataset_i = [ds for ds in self.datasets if ds.split('_')[0] == d]
            for d_i in dataset_i:
                fp_fov = self.find_fp_knn_fov(ct=ct,
                                              dataset_i=d_i,
                                              k=k,
                                              min_count=min_count,
                                              min_support=min_support)
                if len(fp_fov) > 0:
                    fp_d[d_i] = fp_fov

            if len(fp_d) == 1:
                common_patterns = list(fp_d.values())[0]
                common_patterns = common_patterns.rename(columns={'support': f"support_{list(fp_d.keys())[0]}"})
            else:
                comm_fps = set.intersection(*[set(df['itemsets']) for df in
                                              fp_d.values()])  # the items' order in patterns will not affect the returned intersection
                common_patterns = pd.DataFrame({'itemsets': list(comm_fps)})
                for data_name, df in fp_d.items():
                    support_dict = dict(df[['itemsets', 'support']].values)
                    support_dict = {tuple(key): value for key, value in support_dict.items()}
                    common_patterns[f"support_{data_name}"] = common_patterns['itemsets'].apply(
                        lambda x: support_dict.get(tuple(x), None))
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
            sum_rank1 = support_rank[:len(group1)].sum()[0]
            sum_rank2 = support_rank[len(group1):].sum()[0]
            if sum_rank1 > sum_rank2:
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
        return fp_dataset0, fp_dataset1

    def differential_analysis_dist(self,
                                   ct: str,
                                   datasets: List[str],
                                   max_dist: float = 100,
                                   min_support: float = 0.5,
                                   min_size: int = 0,
                                   min_count: int = 0,
                                   ):
        """
        Explore the differences in cell types and frequent patterns of cell types in spatial radius-based neighborhood of cell
        type of interest. Perform differential analysis of frequent patterns in specified datasets.

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
        min_count:
            Minimum number of each cell type to consider.

        Return
        ------
            Dataframes with significant enriched patterns in differential analysis
        """
        if len(datasets) != 2:
            raise ValueError("Require 2 datasets for differential analysis.")
        # Identify frequent patterns for each dataset

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
                                               min_count=min_count,
                                               min_support=min_support)
                if len(fp_fov) > 0:
                    fp_d[d_i] = fp_fov

            if len(fp_d) == 1:
                common_patterns = list(fp_d.values())[0]
                common_patterns = common_patterns.rename(columns={'support': f"support_{list(fp_d.keys())[0]}"})
            else:
                comm_fps = set.intersection(*[set(df['itemsets']) for df in
                                              fp_d.values()])  # the items' order in patterns will not affect the returned intersection
                common_patterns = pd.DataFrame({'itemsets': list(comm_fps)})
                for data_name, df in fp_d.items():
                    support_dict = dict(df[['itemsets', 'support']].values)
                    support_dict = {tuple(key): value for key, value in support_dict.items()}
                    common_patterns[f"support_{data_name}"] = common_patterns['itemsets'].apply(
                        lambda x: support_dict.get(tuple(x), None))
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
            sum_rank1 = support_rank[:len(group1)].sum()[0]
            sum_rank2 = support_rank[len(group1):].sum()[0]
            if sum_rank1 > sum_rank2:
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
        return fp_dataset0, fp_dataset1
