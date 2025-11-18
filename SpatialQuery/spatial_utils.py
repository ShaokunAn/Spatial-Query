"""
Utility functions for spatial_query class.
This module contains helper methods that support the main spatial query operations.
"""

from collections import Counter
from itertools import combinations
from typing import List, Optional, Union, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from mlxtend.frequent_patterns import fpgrowth
from scipy.stats import fisher_exact, percentileofscore
from statsmodels.stats.multitest import multipletests
from scipy.sparse import csr_matrix
from scipy import sparse
import scanpy as sc
from scipy.spatial import KDTree
from scipy import stats as scipy_stats


def initialize_grids(spatial_pos, labels, n_split, overlap_radius):
    """
    Initialize spatial grids for efficient querying.
    
    Parameters
    ----------
    spatial_pos : np.ndarray
        Spatial coordinates of cells
    labels : pd.Series
        Cell type labels
    n_split : int
        Number of splits in each axis
    overlap_radius : float
        Overlap radius for grids
        
    Returns
    -------
    tuple
        (grid_cell_types, grid_indices)
    """
    xmax, ymax = np.max(spatial_pos, axis=0)
    xmin, ymin = np.min(spatial_pos, axis=0)
    x_step = (xmax - xmin) / n_split  # separate x axis into n_split parts
    y_step = (ymax - ymin) / n_split  # separate y axis into n_split parts

    grid_cell_types = {}
    grid_indices = {}

    for i in range(n_split):
        for j in range(n_split):
            x_start = xmin + i * x_step - (overlap_radius if i > 0 else 0)
            x_end = xmin + (i + 1) * x_step + (overlap_radius if i < (n_split - 1) else 0)
            y_start = ymin + j * y_step - (overlap_radius if j > 0 else 0)
            y_end = ymin + (j + 1) * y_step + (overlap_radius if j < (n_split - 1) else 0)

            cell_mask = (spatial_pos[:, 0] >= x_start) & (spatial_pos[:, 0] <= x_end) & \
                        (spatial_pos[:, 1] >= y_start) & (spatial_pos[:, 1] <= y_end)

            grid_indices[(i, j)] = np.where(cell_mask)[0]
            grid_cell_types[(i, j)] = set(labels[cell_mask])

    return grid_cell_types, grid_indices


def query_pattern(pattern, grid_cell_types, grid_indices):
    """
    Query which grids contain all cell types in the pattern.
    
    Parameters
    ----------
    pattern : list
        List of cell types to query
    grid_cell_types : dict
        Dictionary mapping grid coordinates to cell types
    grid_indices : dict
        Dictionary mapping grid coordinates to cell indices
        
    Returns
    -------
    tuple
        (matching_grids, matching_cells_indices)
    """
    matching_grids = []
    matching_cells_indices = {}
    for grid, cell_types in grid_cell_types.items():
        if all(cell_type in cell_types for cell_type in pattern):
            matching_grids.append(grid)
            indices = grid_indices[grid]
            matching_cells_indices[grid] = indices
    return matching_grids, matching_cells_indices


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
    freq_neighbors = Counter(neighbors)
    freq_labels = Counter(labels)
    for element, count in freq_neighbors.items():
        if freq_labels[element] < count:
            return False
    return True


def distinguish_duplicates(transaction: List[str]):
    """
    Append suffix to items of transaction to distinguish the duplicate items.
    """
    counter = dict(Counter(transaction))
    trans_suf = [f"{item}_{i}" for item, value in counter.items() for i in range(value)]
    return trans_suf


def remove_suffix(fp: pd.DataFrame):
    """
    Remove the suffix of frequent patterns.
    """
    trans = [list(tran) for tran in fp['itemsets'].values]
    fp_no_suffix = [[item.split('_')[0] for item in tran] for tran in trans]
    # Create a DataFrame
    fp['itemsets'] = fp_no_suffix
    return fp


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


def build_fptree_dist(kd_tree,
                      labels,
                      cell_pos: np.ndarray = None,
                      spatial_pos: np.ndarray = None,
                      max_dist: float = 100,
                      min_support: float = 0.5,
                      if_max: bool = True,
                      min_size: int = 0,
                      cinds: List[int] = None,
                      max_ns: int = 100) -> tuple:
    """
    Build a frequency pattern tree based on the distance of cell types.

    Parameter
    ---------
    kd_tree:
        KDTree for spatial queries
    labels:
        Cell type labels
    cell_pos:
        Spatial coordinates of input points.
        If cell_pos is None, use all spots in fov to compute frequent patterns.
    spatial_pos:
        All spatial positions (used if cell_pos is None)
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
    A tuple containing the FPs, the transactions table and the neighbors index.
    """
    if cell_pos is None:
        cell_pos = spatial_pos

    idxs = kd_tree.query_ball_point(cell_pos, r=max_dist, return_sorted=False, workers=-1)
    if cinds is None:
        cinds = list(range(len(idxs)))

    # Prepare data for FP-Tree construction
    transactions = []
    valid_idxs = []
    labels_array = np.array(labels)
    for i_idx, idx in zip(cinds, idxs):
        if not idx:
            continue
        idx_array = np.array(idx)
        valid_mask = idx_array != i_idx
        valid_indices = idx_array[valid_mask][:max_ns]

        transaction = labels_array[valid_indices]
        if len(transaction) > min_size:
            transactions.append(transaction.tolist())
            valid_idxs.append(valid_indices)

    # Convert transactions to a DataFrame suitable for fpgrowth
    mlb = MultiLabelBinarizer()
    encoded_data = mlb.fit_transform(transactions)
    df = pd.DataFrame(encoded_data.astype(bool), columns=mlb.classes_)

    # Construct FP-Tree using fpgrowth
    fp_tree = fpgrowth(df, min_support=min_support, use_colnames=True)

    if if_max:
        fp_tree = find_maximal_patterns(fp=fp_tree)

    if len(fp_tree) == 0:
        return pd.DataFrame(columns=['support', 'itemsets']), df, valid_idxs
    else:
        fp_tree['itemsets'] = fp_tree['itemsets'].apply(lambda x: tuple(sorted(x)))
        fp_tree = fp_tree.drop_duplicates().reset_index(drop=True)
        fp_tree['itemsets'] = fp_tree['itemsets'].apply(lambda x: list(x))
        fp_tree = fp_tree.sort_values(by='support', ignore_index=True, ascending=False)
        return fp_tree, df, valid_idxs


def build_fptree_knn(kd_tree,
                     labels,
                     cell_pos: np.ndarray = None,
                     spatial_pos: np.ndarray = None,
                     k: int = 30,
                     min_support: float = 0.5,
                     max_dist: float = 200,
                     if_max: bool = True
                     ) -> tuple:
    """
    Build a frequency pattern tree based on knn

    Parameter
    ---------
    kd_tree:
        KDTree for spatial queries
    labels:
        Cell type labels
    cell_pos:
        Spatial coordinates of input points.
        If cell_pos is None, use all spots in fov to compute frequent patterns.
    spatial_pos:
        All spatial positions (used if cell_pos is None)
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
        cell_pos = spatial_pos

    dists, idxs = kd_tree.query(cell_pos, k=k + 1, workers=-1)

    # Prepare data for FP-Tree construction
    idxs = np.array(idxs)
    dists = np.array(dists)
    labels_array = np.array(labels)
    transactions = []
    mask = dists < max_dist
    for i, idx in enumerate(idxs):
        inds = idx[mask[i]]
        if len(inds) == 0:
            continue
        transaction = labels_array[inds[1:]]
        transactions.append(transaction)

    # Convert transactions to a DataFrame suitable for fpgrowth
    mlb = MultiLabelBinarizer()
    encoded_data = mlb.fit_transform(transactions)
    df = pd.DataFrame(encoded_data.astype(bool), columns=mlb.classes_)

    # Construct FP-Tree using fpgrowth
    fp_tree = fpgrowth(df, min_support=min_support, use_colnames=True)

    if if_max:
        fp_tree = find_maximal_patterns(fp_tree)

    if len(fp_tree) == 0:
        return pd.DataFrame(columns=['support', 'itemsets']), df, idxs
    else:
        fp_tree['itemsets'] = fp_tree['itemsets'].apply(lambda x: tuple(sorted(x)))
        fp_tree = fp_tree.drop_duplicates().reset_index(drop=True)
        fp_tree['itemsets'] = fp_tree['itemsets'].apply(lambda x: list(x))
        fp_tree = fp_tree.sort_values(by='support', ignore_index=True, ascending=False)
        return fp_tree, df, idxs


def de_genes_scanpy(adata,
                    genes_list,
                    ind_group1: Union[List[int], np.ndarray],
                    ind_group2: Union[List[int], np.ndarray],
                    genes: Optional[Union[str, List[str]]] = None,
                    min_fraction: float = 0.05,
                    method: str = 't-test',
                    ) -> pd.DataFrame:
    """
    Perform differential expression analysis using scanpy's rank_genes_groups.
    Supports t-test and wilcoxon methods.
    """

    if genes_list is None or len(genes_list) != adata.n_vars:
        raise ValueError("genes_list is None or does not match adata.n_vars.")
    
    # Convert indices to numpy arrays
    ind_group1 = np.array(ind_group1, dtype=np.int32)
    ind_group2 = np.array(ind_group2, dtype=np.int32)
    
    n1, n2 = len(ind_group1), len(ind_group2)

    if isinstance(adata.X, csr_matrix):
        X1 = adata.X[ind_group1, :]
        X2 = adata.X[ind_group2, :]
        # Count expressing cells efficiently on sparse matrix
        count1 = np.asarray((X1 > 0).sum(axis=0)).flatten()
        count2 = np.asarray((X2 > 0).sum(axis=0)).flatten()
    else:
        X1 = adata.X[ind_group1, :]
        X2 = adata.X[ind_group2, :]
        # Count expressing cells
        count1 = (X1 > 0).sum(axis=0)
        count2 = (X2 > 0).sum(axis=0)
        if hasattr(count1, 'A1'):  # Handle matrix type
            count1 = count1.A1
            count2 = count2.A1
    prop1 = count1 / n1 
    prop2 = count2 / n2 

    pre_mask = (prop1 >= min_fraction) | (prop2 >= min_fraction)

    # Create temporary group labels
    group_labels = np.array(['other'] * adata.n_obs, dtype=object)
    group_labels[ind_group1] = 'group1'
    group_labels[ind_group2] = 'group2'
    
    # Create a temporary copy with group labels
    adata_temp = adata.copy()
    adata_temp.obs['_temp_de_group'] = pd.Categorical(group_labels)
    
    # Filter to only cells in the two groups
    adata_temp = adata_temp[adata_temp.obs['_temp_de_group'] != 'other']
    
    # Apply pre_mask filtering to genes based on min_fraction
    genes_to_test = np.array(genes_list)[pre_mask].tolist()
    adata_temp = adata_temp[:, genes_to_test]
    
    # If specific genes requested, further subset adata to only those genes
    if genes is not None:
        if isinstance(genes, str):
            genes = [genes]
        
        # Find valid genes that exist in the filtered genes
        valid_gene_names = [g for g in genes if g in genes_to_test]
        invalid_genes = [g for g in genes if g not in genes_to_test]
        
        if len(invalid_genes) > 0:
            print(f'Invalid genes {invalid_genes} will be skipped.')
        
        if len(valid_gene_names) == 0:
            print('No valid genes found. Returning empty DataFrame.')
            return pd.DataFrame()
        
        # Subset to valid genes using gene names
        adata_temp = adata_temp[:, valid_gene_names]
        print(f'Testing {len(valid_gene_names)} specified genes ...')
    else:
        print(f'Testing {len(genes_to_test)} genes ...')
    
    
    # Run scanpy's differential expression
    sc.tl.rank_genes_groups(
        adata_temp,
        groupby='_temp_de_group',
        method=method,
        use_raw=False,
        pts=True,  # Calculate proportion of cells expressing each gene
        tie_correct=True if method == 'wilcoxon' else False,
    )
    
    # Extract results
    result = sc.get.rank_genes_groups_df(adata_temp, group='group1')
    
    # Rename and reformat columns
    result_df = pd.DataFrame({
        'gene': result['names'].values,
        'p_value': result['pvals'].values,
        'adj_p_value': result['pvals_adj'].values,
        'log2fc': result['logfoldchanges'].values,
    })
    
    # Extract proportion data
    result_df['proportion_1'] = result['pct_nz_group'].values
    result_df['proportion_2'] = result['pct_nz_reference'].values 
    
    result_df['abs_difference'] = np.abs(result_df['proportion_1'] - result_df['proportion_2'])
    
    # Determine DE direction
    result_df['de_in'] = np.where(
        result_df['log2fc'] > 0, 'group1',
        np.where(result_df['log2fc'] < 0, 'group2', None)
    )
    
    # Filter by adjusted p-value and sort
    result_df = result_df[result_df['adj_p_value'] < 0.05].sort_values('p_value').reset_index(drop=True)
    
    return result_df


def de_genes_fisher(adata,
                    genes_list,
                    ind_group1: Union[List[int], np.ndarray],
                    ind_group2: Union[List[int], np.ndarray],
                    genes: Optional[Union[str, List[str]]] = None,
                    min_fraction: float = 0.05,
                    ) -> pd.DataFrame:
    """
    Perform Fisher's exact test using adata.X directly. This method is used when build_gene_index=False.
    Optimized to work with sparse matrices and memory-efficient.
    """
    
    # Ensure we use the correct gene names that match adata.X columns
    if genes_list is None or len(genes_list) != adata.n_vars:
        raise ValueError("genes_list is None or does not match adata.n_vars.")
    
    # Convert indices to numpy arrays
    ind_group1 = np.array(ind_group1, dtype=np.int32)
    ind_group2 = np.array(ind_group2, dtype=np.int32)
    
    n1, n2 = len(ind_group1), len(ind_group2)
    
    # Count expressing cells directly from sparse matrix (memory-efficient)
    if isinstance(adata.X, csr_matrix):
        X1 = adata.X[ind_group1, :]
        X2 = adata.X[ind_group2, :]
        # Count expressing cells efficiently on sparse matrix
        count1 = np.asarray((X1 > 0).sum(axis=0)).flatten()
        count2 = np.asarray((X2 > 0).sum(axis=0)).flatten()
    else:
        X1 = adata.X[ind_group1, :]
        X2 = adata.X[ind_group2, :]
        # Count expressing cells
        count1 = (X1 > 0).sum(axis=0)
        count2 = (X2 > 0).sum(axis=0)
        if hasattr(count1, 'A1'):  # Handle matrix type
            count1 = count1.A1
            count2 = count2.A1
    
    # Calculate proportions of expressing cells for each gene
    prop1 = count1 / n1 
    prop2 = count2 / n2
    
    # Pre-filter genes based on min_fraction before Fisher's test
    pre_mask = (prop1 >= min_fraction) | (prop2 >= min_fraction)
    
    # IMPORTANT: Filter genes BEFORE testing to ensure correct multiple testing correction
    # If specific genes requested, filter to those
    if genes is not None:
        if isinstance(genes, str):
            genes = [genes]
        gene_mask = np.isin(genes_list, genes)
        invalid_genes = [g for g in genes if g not in genes_list]
        if len(invalid_genes) > 0:
            print(f'Invalid genes {invalid_genes} will be skipped.')
        if np.sum(gene_mask) == 0:
            print('No valid genes found. Returning empty DataFrame.')
            return pd.DataFrame()
        
        pre_mask &= gene_mask
    
    # Get indices of genes to test
    gene_indices = np.where(pre_mask)[0]
    n_genes_to_test = len(gene_indices)
    print(f'Testing {n_genes_to_test} genes with Fisher\'s exact test ...')
    
    if n_genes_to_test == 0:
        print('All genes are below the minimum fraction. Returning empty DataFrame.')
        return pd.DataFrame()
    
    # Prepare contingency table parameters (vectorized)
    # Fisher's exact test contingency table:
    #                 | Expressed | Not Expressed |
    # Group1          |     a     |       b       |
    # Group2          |     c     |       d       |
    a = count1[gene_indices].astype(int)  # expressed in group1
    b = (n1 - a).astype(int)              # not expressed in group1
    c = count2[gene_indices].astype(int)  # expressed in group2
    d = (n2 - c).astype(int)              # not expressed in group2
    
    # Compute Fisher's exact test for each gene
    # Note: Fisher's exact test cannot be fully vectorized, must loop
    p_values = []
    for i in range(n_genes_to_test):
        # Construct 2x2 contingency table
        table = [[int(a[i]), int(b[i])], 
                [int(c[i]), int(d[i])]]
        _, p_value = fisher_exact(table, alternative='two-sided')
        p_values.append(p_value)
    
    p_values = np.array(p_values)
    
    # Build results DataFrame
    result_df = pd.DataFrame({
        'gene': np.array(genes_list)[gene_indices],  # Use correct gene names
        'proportion_1': prop1[gene_indices],
        'proportion_2': prop2[gene_indices],
        'abs_difference': np.abs(prop1[gene_indices] - prop2[gene_indices]),
        'p_value': p_values,
        'expressed_cells_1': count1[gene_indices].astype(int),
        'expressed_cells_2': count2[gene_indices].astype(int),
        'total_cells_1': n1,
        'total_cells_2': n2,
    })
    
    # Filter out invalid p-values
    result_df = result_df[~np.isnan(result_df['p_value'])]
    
    if len(result_df) == 0:
        return pd.DataFrame()
    
    # Multiple testing correction (FDR-BH method)
    result_df['adj_p_value'] = multipletests(result_df['p_value'], method='fdr_bh')[1]
    
    # Determine DE direction based on proportion difference
    result_df['de_in'] = np.where(
        (result_df['proportion_1'] >= result_df['proportion_2']),
        'group1',
        np.where(
            (result_df['proportion_2'] > result_df['proportion_1']),
            'group2',
            None
        )
    )
    
    # Filter by adjusted p-value and sort by p-value
    result_df = result_df[result_df['adj_p_value'] < 0.05].sort_values('p_value').reset_index(drop=True)
    
    return result_df

def _auto_normalize_spatial_coords(spatial_coords: np.ndarray)->np.ndarray:
    """
    Normalize spatial coordinates to have a mean nearest neighbor distance of 1.

    Parameters
    ----------
    spatial_coords : np.ndarray
        Spatial coordinates of cells.

    Returns
    -------
    np.ndarray
        Normalized spatial coordinates.
    """
    kd_tree = KDTree(spatial_coords)
    distances, _ = kd_tree.query(spatial_coords, k=2)
    nn_dist = np.mean(distances[:, 1])
    scale_factor = 1.0 / nn_dist
    spatial_pos_norm = spatial_coords * scale_factor

    print(f"\nAuto-normalizing spatial coordinates: mean nearest neighbor distance = 1.0")
    print(f"Scale factor: {scale_factor:.4f}")

    return spatial_pos_norm


def get_motif_neighbor_cells(sq_obj,
                             ct: str,
                             motif: Union[List[str], str],
                             max_dist: Optional[float] = None,
                             k: Optional[int] = None,
                             min_size: int = 0) -> dict:
    """
    Get cell IDs of motif cells that are neighbors of the center cell type.
    Similar to motif_enrichment_* with return_cellID=True, but only returns cell IDs
    and move center type of motifs to center cells for gene-gene covarying analysis.

    For kNN: filters out neighbors beyond min(max_dist, sq_obj.max_radius).
    For dist: filters out center cells with fewer than min_size neighbors.

    If center type (ct) is in motif, motif cells of center type are also included in center_id.

    Parameter
    ---------
    sq_obj : spatial_query object
        The spatial_query object
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

    motif_exc = [m for m in motif if m not in sq_obj.labels.unique()]
    if len(motif_exc) != 0:
        print(f"Found no {motif_exc} in {sq_obj.label_key}. Ignoring them.")
    motif = [m for m in motif if m not in motif_exc]

    if ct not in sq_obj.labels.unique():
        raise ValueError(f"Found no {ct} in {sq_obj.label_key}!")

    cinds = np.where(sq_obj.labels == ct)[0]

    # Check if ct is in motif - if so, we'll add those motif cells to center_id later
    ct_in_motif = ct in motif

    motif_mask = np.isin(np.array(sq_obj.labels), motif)

    if max_dist is not None:
        # Distance-based neighbors - only query for center cells
        max_dist = min(max_dist, sq_obj.max_radius)
        idxs_centers = sq_obj.kd_tree.query_ball_point(
            sq_obj.spatial_pos[cinds],
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
                neighbor_labels = sq_obj.labels[neighbors]
                neighbors_unique = neighbor_labels.unique().tolist()
                has_all_motifs = all([m in neighbors_unique for m in motif])

                if has_all_motifs:
                    # Get motif neighbors for this center
                    motif_neighbors = neighbors[motif_mask[neighbors]]

                    # Build pairs in the same loop (vectorized)
                    if ct_in_motif:
                        center_type_neighs = motif_neighbors[sq_obj.labels[motif_neighbors] == ct]
                        non_center_type_neighs = motif_neighbors[sq_obj.labels[motif_neighbors] != ct]

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
        dists, idxs = sq_obj.kd_tree.query(sq_obj.spatial_pos[cinds], k=k + 1, workers=-1)

        # Apply distance cutoff
        if max_dist is None:
            max_dist = sq_obj.max_radius
        max_dist = min(max_dist, sq_obj.max_radius)

        valid_neighbors = dists[:, 1:] <= max_dist

        # Process all centers in one loop and build pairs
        center_neighbor_pairs = []

        for i, cind in enumerate(cinds):
            valid_neighs = idxs[i, 1:][valid_neighbors[i, :]]

            # Check if all motif types are present in neighbors
            neighbor_labels = sq_obj.labels[valid_neighs]
            neighbors_unique = neighbor_labels.unique().tolist()
            has_all_motifs = all([m in neighbors_unique for m in motif])

            if has_all_motifs:
                # Get motif neighbors for this center
                motif_neighbors = valid_neighs[motif_mask[valid_neighs]]

                # Build pairs in the same loop (vectorized)
                if ct_in_motif:
                    center_type_neighs = motif_neighbors[sq_obj.labels[motif_neighbors] == ct]
                    non_center_type_neighs = motif_neighbors[sq_obj.labels[motif_neighbors] != ct]

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


def get_all_neighbor_cells(sq_obj,
                           ct: str,
                           max_dist: Optional[float] = None,
                           k: Optional[int] = None,
                           min_size: int = 0,
                           exclude_centers: Optional[np.ndarray] = None,
                           exclude_neighbors: Optional[np.ndarray] = None) -> dict:
    """
    Get all neighbor cells (not limited to motif) for given center cell type excluding center cells in exclude_centers.
    Similar to get_motif_neighbor_cells but returns ALL neighbors regardless of cell type.
    And only returns neighbors that are different from center cell type.

    Parameter
    ---------
    sq_obj : spatial_query object
        The spatial_query object
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

    if ct not in sq_obj.labels.unique():
        raise ValueError(f"Found no {ct} in {sq_obj.label_key}!")

    cinds = np.where(sq_obj.labels == ct)[0]

    # Exclude specified centers if provided
    if exclude_centers is not None:
        cinds = np.setdiff1d(cinds, exclude_centers)

    # Build pairs directly
    center_neighbor_pairs = []

    if max_dist is not None:
        # Distance-based neighbors - only query for center cells
        max_dist = min(max_dist, sq_obj.max_radius)
        idxs_centers = sq_obj.kd_tree.query_ball_point(
            sq_obj.spatial_pos[cinds],
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
                valid_neighbors = neighbors[sq_obj.labels[neighbors] != ct]

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
        dists, idxs = sq_obj.kd_tree.query(sq_obj.spatial_pos[cinds], k=k + 1, workers=-1)

        # Apply distance cutoff
        if max_dist is None:
            max_dist = sq_obj.max_radius
        max_dist = min(max_dist, sq_obj.max_radius)

        valid_neighbors = dists[:, 1:] <= max_dist

        # Process all centers in one loop and build pairs
        for i, cind in enumerate(cinds):
            valid_neighs = idxs[i, 1:][valid_neighbors[i, :]]

            # Exclude center type from neighbors
            valid_neighs_filtered = valid_neighs[sq_obj.labels[valid_neighs] != ct]

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


