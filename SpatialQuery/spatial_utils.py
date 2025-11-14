"""
Utility functions for spatial_query class.
This module contains helper methods that support the main spatial query operations.
"""

from collections import Counter
from itertools import combinations
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from mlxtend.frequent_patterns import fpgrowth
from scipy.stats import fisher_exact
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


def compute_covariance_statistics_paired(expr_genes,
                                         pair_centers,
                                         pair_neighbors,
                                         center_mean,
                                         cell_type_means,
                                         neighbor_cell_types,
                                         is_sparse):
    """
    Compute raw covariance statistics (cov_sum, center_ss, neighbor_ss) for paired data.
    This version is optimized for sparse matrices and designed for aggregating across FOVs.

    Parameters
    ----------
    expr_genes : sparse or dense matrix
        Gene expression matrix (n_cells × n_genes)
    pair_centers : array
        Indices of center cells in each pair
    pair_neighbors : array
        Indices of neighbor cells in each pair
    center_mean : array
        Mean expression for center cell type (n_genes,)
    cell_type_means : dict
        Dictionary mapping cell types to their mean expression (n_genes,)
    neighbor_cell_types : array
        Cell type labels for each neighbor in pair_neighbors
    is_sparse : bool
        Whether expr_genes is sparse

    Returns
    -------
    cov_sum : ndarray
        Sum of centered cross products (n_genes × n_genes)
    center_ss : ndarray
        Sum of squared centered center expression (n_genes,)
    neighbor_ss : ndarray
        Sum of squared centered neighbor expression (n_genes,)
    n_pairs : int
        Number of pairs
    n_eff : int
        Effective sample size (min of unique centers and unique neighbors)
    """
    pair_center_expr = expr_genes[pair_centers, :]
    pair_neighbor_expr = expr_genes[pair_neighbors, :]

    n_pairs = len(pair_centers)
    n_eff = min(len(np.unique(pair_centers)), len(np.unique(pair_neighbors)))

    if is_sparse:
        # Get unique neighbor types and their counts
        unique_neighbor_types, type_counts = np.unique(neighbor_cell_types, return_counts=True)
        n_types = len(unique_neighbor_types)

        # Build mapping for creating indicator matrix
        type_to_idx = {ct: idx for idx, ct in enumerate(unique_neighbor_types)}
        type_indices = np.array([type_to_idx[ct] for ct in neighbor_cell_types])

        # Stack cell-type-specific means into a matrix (n_genes × n_types)
        neighbor_type_means_matrix = np.column_stack([cell_type_means[ct] for ct in unique_neighbor_types])

        # ==================== Covariance sum computation ====================
        # Term 1: Σ_i x_i * y_i
        cross_product = pair_center_expr.T @ pair_neighbor_expr
        if sparse.issparse(cross_product):
            cross_product = np.asarray(cross_product.todense())

        # Term 2: -Σ_i x_i * μ^{ct_i}
        type_indicator = csr_matrix((np.ones(n_pairs), (np.arange(n_pairs), type_indices)),
                                   shape=(n_pairs, n_types))

        sum_center_by_type = pair_center_expr.T @ type_indicator
        if sparse.issparse(sum_center_by_type):
            sum_center_by_type = np.asarray(sum_center_by_type.todense())

        term2 = sum_center_by_type @ neighbor_type_means_matrix.T

        # Term 3: -μ_X * Σ_i y_i
        sum_neighbor = np.array(pair_neighbor_expr.sum(axis=0)).flatten()
        term3 = np.outer(center_mean, sum_neighbor)

        # Term 4: μ_X * Σ_i μ^{ct_i}
        weighted_neighbor_mean = neighbor_type_means_matrix @ type_counts
        term4 = np.outer(center_mean, weighted_neighbor_mean)

        cov_sum = cross_product - term2 - term3 + term4

        # ==================== Sum of squares computation ====================
        # Center: Σ_i (x_i - μ_X)^2 = Σ_i x_i^2 - 2*μ_X*Σ_i x_i + n*μ_X^2
        sum_sq_center = np.array(pair_center_expr.power(2).sum(axis=0)).flatten()
        sum_center = np.array(pair_center_expr.sum(axis=0)).flatten()
        center_ss = sum_sq_center - 2 * center_mean * sum_center + n_pairs * center_mean**2

        # Neighbor: Σ_i (y_i - μ^{ct_i})^2
        sum_sq_neighbor = np.array(pair_neighbor_expr.power(2).sum(axis=0)).flatten()

        sum_neighbor_by_type = pair_neighbor_expr.T @ type_indicator
        if sparse.issparse(sum_neighbor_by_type):
            sum_neighbor_by_type = np.asarray(sum_neighbor_by_type.todense())

        term_y_mean = (sum_neighbor_by_type * neighbor_type_means_matrix).sum(axis=1)
        term_mean_sq = (neighbor_type_means_matrix**2) @ type_counts

        neighbor_ss = sum_sq_neighbor - 2 * term_y_mean + term_mean_sq

    else:
        # Dense matrix operations
        neighbor_type_means_matrix = np.array([cell_type_means[ct] for ct in neighbor_cell_types])

        pair_center_shifted = pair_center_expr - center_mean[np.newaxis, :]
        pair_neighbor_shifted = pair_neighbor_expr - neighbor_type_means_matrix

        # Compute covariance sum
        cov_sum = pair_center_shifted.T @ pair_neighbor_shifted

        # Compute sum of squares
        center_ss = (pair_center_shifted**2).sum(axis=0)
        neighbor_ss = (pair_neighbor_shifted**2).sum(axis=0)

    return cov_sum, center_ss, neighbor_ss, n_pairs, n_eff


def compute_covariance_statistics_all_to_all(expr_genes,
                                               center_cells,
                                               neighbor_cells,
                                               center_mean,
                                               cell_type_means,
                                               neighbor_cell_types,
                                               is_sparse):
    """
    Compute raw covariance statistics (cov_sum, center_ss, neighbor_ss) for all-to-all pairs.
    This version is optimized for sparse matrices and designed for aggregating across FOVs.

    Parameters
    ----------
    expr_genes : sparse or dense matrix
        Gene expression matrix (n_cells × n_genes)
    center_cells : array
        Indices of center cells
    neighbor_cells : array
        Indices of neighbor cells
    center_mean : array
        Mean expression for center cell type (n_genes,)
    cell_type_means : dict
        Dictionary mapping cell types to their mean expression (n_genes,)
    neighbor_cell_types : array
        Cell type labels for each cell in neighbor_cells
    is_sparse : bool
        Whether expr_genes is sparse

    Returns
    -------
    cov_sum : ndarray
        Sum of centered cross products (n_genes × n_genes)
    center_ss : ndarray
        Sum of squared centered center expression (n_genes,)
    neighbor_ss : ndarray
        Sum of squared centered neighbor expression (n_genes,)
    n_pairs : int
        Number of pairs (len(center_cells) * len(neighbor_cells))
    n_eff : int
        Effective sample size (min of n_centers and n_neighbors)
    """
    center_expr = expr_genes[center_cells, :]
    neighbor_expr = expr_genes[neighbor_cells, :]

    n_centers = len(center_cells)
    n_neighbors = len(neighbor_cells)
    n_pairs = n_centers * n_neighbors
    n_eff = min(n_centers, n_neighbors)
    n_genes = expr_genes.shape[1]

    if is_sparse:
        # Get unique neighbor types and their counts
        unique_neighbor_types, type_counts = np.unique(neighbor_cell_types, return_counts=True)
        n_types = len(unique_neighbor_types)

        # Build mapping for neighbor types
        type_to_idx = {ct: idx for idx, ct in enumerate(unique_neighbor_types)}
        type_indices = np.array([type_to_idx[ct] for ct in neighbor_cell_types])

        # Stack cell-type-specific means into a matrix (n_genes × n_types)
        neighbor_type_means_matrix = np.column_stack([cell_type_means[ct] for ct in unique_neighbor_types])

        # ==================== Covariance sum computation ====================
        # Term 1: Σ_i Σ_j x_i * y_j = (Σ_i x_i) * (Σ_j y_j)
        sum_center = np.array(center_expr.sum(axis=0)).flatten()
        sum_neighbor = np.array(neighbor_expr.sum(axis=0)).flatten()
        term1 = np.outer(sum_center, sum_neighbor)

        # Term 2: -Σ_i Σ_j x_i * μ^{ct_j} = (Σ_i x_i) * (Σ_j μ^{ct_j})
        weighted_neighbor_mean = neighbor_type_means_matrix @ type_counts
        term2 = np.outer(sum_center, weighted_neighbor_mean)

        # Term 3: -Σ_i Σ_j μ_X * y_j = n_centers * μ_X * (Σ_j y_j)
        term3 = n_centers * np.outer(center_mean, sum_neighbor)

        # Term 4: Σ_i Σ_j μ_X * μ^{ct_j} = n_centers * μ_X * (Σ_j μ^{ct_j})
        term4 = n_centers * np.outer(center_mean, weighted_neighbor_mean)

        cov_sum = term1 - term2 - term3 + term4

        # ==================== Sum of squares computation ====================
        # Center: Σ_i Σ_j (x_i - μ_X)^2 = n_neighbors * Σ_i (x_i - μ_X)^2
        sum_sq_center = np.array(center_expr.power(2).sum(axis=0)).flatten()
        center_ss = n_neighbors * (sum_sq_center - 2 * center_mean * sum_center + n_centers * center_mean**2)

        # Neighbor: Σ_i Σ_j (y_j - μ^{ct_j})^2 = n_centers * Σ_j (y_j - μ^{ct_j})^2
        sum_sq_neighbor = np.array(neighbor_expr.power(2).sum(axis=0)).flatten()

        type_indicator = csr_matrix((np.ones(n_neighbors), (np.arange(n_neighbors), type_indices)),
                                   shape=(n_neighbors, n_types))

        sum_neighbor_by_type = neighbor_expr.T @ type_indicator
        if sparse.issparse(sum_neighbor_by_type):
            sum_neighbor_by_type = np.asarray(sum_neighbor_by_type.todense())

        term_y_mean = (sum_neighbor_by_type * neighbor_type_means_matrix).sum(axis=1)
        term_mean_sq = (neighbor_type_means_matrix**2) @ type_counts

        neighbor_ss = n_centers * (sum_sq_neighbor - 2 * term_y_mean + term_mean_sq)

    else:
        # Dense matrix operations
        center_shifted = center_expr - center_mean[np.newaxis, :]

        neighbor_type_means_matrix = np.array([cell_type_means[ct] for ct in neighbor_cell_types])
        neighbor_shifted = neighbor_expr - neighbor_type_means_matrix

        # Compute covariance sum: Σ_i Σ_j x'_i * y'_j = (Σ_i x'_i) @ (Σ_j y'_j).T
        sum_center_shifted = center_shifted.sum(axis=0)
        sum_neighbor_shifted = neighbor_shifted.sum(axis=0)
        cov_sum = np.outer(sum_center_shifted, sum_neighbor_shifted)

        # Compute sum of squares
        center_ss = n_neighbors * (center_shifted**2).sum(axis=0)
        neighbor_ss = n_centers * (neighbor_shifted**2).sum(axis=0)

    return cov_sum, center_ss, neighbor_ss, n_pairs, n_eff


def compute_cross_correlation_paired(sq_obj,
                                     expr_genes,
                                     pair_centers,
                                     pair_neighbors,
                                     center_mean,
                                     cell_type_means,
                                     neighbor_cell_types,
                                     is_sparse):
    """
    Helper function to compute cross-correlation for paired data.
    Used in gene-gene correlation analysis when center-neighbor pairs are pre-defined.

    Parameters
    ----------
    sq_obj : spatial_query object
        The spatial_query object (for accessing labels if needed)
    expr_genes : sparse or dense matrix
        Gene expression matrix (n_cells × n_genes)
    pair_centers : array
        Indices of center cells in each pair
    pair_neighbors : array
        Indices of neighbor cells in each pair
    center_mean : array
        Global mean expression for center cell type (n_genes,)
    cell_type_means : dict
        Dictionary mapping cell types to their global mean expression
    neighbor_cell_types : array
        Cell type labels for each neighbor cell in pairs
    is_sparse : bool
        Whether expr_genes is sparse

    Returns
    -------
    corr_matrix : array
        Correlation matrix (n_genes × n_genes)
    n_eff : int
        Effective sample size
    """
    from scipy import sparse

    # Extract paired expression data
    pair_center_expr = expr_genes[pair_centers, :]
    pair_neighbor_expr = expr_genes[pair_neighbors, :]

    n_pairs = len(pair_centers)
    n_genes = expr_genes.shape[1]

    if is_sparse:
        # Get unique neighbor types and their counts
        unique_neighbor_types, type_counts = np.unique(neighbor_cell_types, return_counts=True)
        n_types = len(unique_neighbor_types)

        # Build mapping for creating indicator matrix
        type_to_idx = {ct: idx for idx, ct in enumerate(unique_neighbor_types)}
        type_indices = np.array([type_to_idx[ct] for ct in neighbor_cell_types])

        # Stack cell-type-specific means into a matrix (n_genes × n_types)
        neighbor_type_means_matrix = np.column_stack([cell_type_means[ct] for ct in unique_neighbor_types])

        # ==================== Cross-covariance computation ====================
        # Term 1: (1/n) Σ_i x_i * y_i
        cross_product = pair_center_expr.T @ pair_neighbor_expr
        if sparse.issparse(cross_product):
            cross_product = np.asarray(cross_product.todense())
        term1 = cross_product / n_pairs

        # Term 2: -(1/n) Σ_i x_i * μ^{ct_i}
        type_indicator = csr_matrix((np.ones(n_pairs), (np.arange(n_pairs), type_indices)),
                                   shape=(n_pairs, n_types))

        sum_center_by_type = pair_center_expr.T @ type_indicator
        if sparse.issparse(sum_center_by_type):
            sum_center_by_type = np.asarray(sum_center_by_type.todense())

        term2 = (sum_center_by_type @ neighbor_type_means_matrix.T) / n_pairs

        # Term 3: -(μ_X/n) Σ_i y_i
        sum_neighbor = np.array(pair_neighbor_expr.sum(axis=0)).flatten()
        term3 = np.outer(center_mean, sum_neighbor / n_pairs)

        # Term 4: (μ_X/n) Σ_i μ^{ct_i}
        weighted_neighbor_mean = (neighbor_type_means_matrix @ type_counts) / n_pairs
        term4 = np.outer(center_mean, weighted_neighbor_mean)

        cross_cov = term1 - term2 - term3 + term4

        # ==================== Variance computation ====================
        # Var(X - μ_X)
        sum_sq_center = np.array(pair_center_expr.power(2).sum(axis=0)).flatten()
        sum_center = np.array(pair_center_expr.sum(axis=0)).flatten()
        var_center = (sum_sq_center / n_pairs
                     - 2 * center_mean * sum_center / n_pairs
                     + center_mean**2)

        # Var(Y - μ^{ct_i})
        sum_sq_neighbor = np.array(pair_neighbor_expr.power(2).sum(axis=0)).flatten()

        sum_neighbor_by_type = pair_neighbor_expr.T @ type_indicator
        if sparse.issparse(sum_neighbor_by_type):
            sum_neighbor_by_type = np.asarray(sum_neighbor_by_type.todense())

        term_y_mean = (sum_neighbor_by_type * neighbor_type_means_matrix).sum(axis=1) / n_pairs
        term_mean_sq = ((neighbor_type_means_matrix**2) @ type_counts) / n_pairs

        var_neighbor = sum_sq_neighbor / n_pairs - 2 * term_y_mean + term_mean_sq

        std_center = np.sqrt(np.maximum(var_center, 0))
        std_neighbor = np.sqrt(np.maximum(var_neighbor, 0))
    else:
        # Dense matrix operations
        neighbor_type_means_matrix = np.array([cell_type_means[ct] for ct in neighbor_cell_types])

        pair_center_shifted = pair_center_expr - center_mean[np.newaxis, :]
        pair_neighbor_shifted = pair_neighbor_expr - neighbor_type_means_matrix

        # Compute covariance matrix
        cross_cov = (pair_center_shifted.T @ pair_neighbor_shifted) / n_pairs

        # Compute standard deviations
        std_center = np.sqrt(np.maximum((pair_center_shifted**2).sum(axis=0) / n_pairs, 0))
        std_neighbor = np.sqrt(np.maximum((pair_neighbor_shifted**2).sum(axis=0) / n_pairs, 0))

    # Correlation matrix
    std_outer = np.outer(std_center, std_neighbor)
    std_outer[std_outer == 0] = 1e-10

    corr_matrix = cross_cov / std_outer

    # Effective sample size
    center_unique = len(np.unique(pair_centers))
    neighbor_unique = len(np.unique(pair_neighbors))
    n_eff = min(center_unique, neighbor_unique)

    return corr_matrix, n_eff


def compute_cross_correlation_all_to_all(sq_obj,
                                         expr_genes,
                                         center_cells,
                                         non_neighbor_cells,
                                         center_mean,
                                         cell_type_means,
                                         non_neighbor_cell_types,
                                         is_sparse):
    """
    Helper function to compute cross-correlation for all-to-all pairs.
    Used in gene-gene correlation analysis when comparing center cells with distant cells.

    Parameters
    ----------
    sq_obj : spatial_query object
        The spatial_query object (for accessing labels if needed)
    expr_genes : sparse or dense matrix
        Gene expression matrix (n_cells × n_genes)
    center_cells : array
        Indices of center cells
    non_neighbor_cells : array
        Indices of non-neighbor cells
    center_mean : array
        Global mean expression for center cell type (n_genes,)
    cell_type_means : dict
        Dictionary mapping cell types to their global mean expression
    non_neighbor_cell_types : array
        Cell type labels for non-neighbor cells
    is_sparse : bool
        Whether expr_genes is sparse

    Returns
    -------
    corr_matrix : array
        Correlation matrix (n_genes × n_genes)
    n_eff : int
        Effective sample size
    """
    from scipy import sparse

    center_expr = expr_genes[center_cells, :]
    non_neighbor_expr = expr_genes[non_neighbor_cells, :]

    n_center = len(center_cells)
    n_non_neighbor = len(non_neighbor_cells)
    n_genes = expr_genes.shape[1]

    if is_sparse and sparse.issparse(center_expr):
        # Get unique non-neighbor types and their counts
        unique_non_neighbor_types, type_counts = np.unique(non_neighbor_cell_types, return_counts=True)
        n_types = len(unique_non_neighbor_types)

        # Build mapping for creating indicator matrix
        type_to_idx = {ct: idx for idx, ct in enumerate(unique_non_neighbor_types)}
        type_indices = np.array([type_to_idx[ct] for ct in non_neighbor_cell_types])

        # Stack cell-type-specific means (n_genes × n_types)
        non_neighbor_type_means_matrix = np.column_stack([cell_type_means[ct] for ct in unique_non_neighbor_types])

        # Compute center statistics
        sum_center = np.array(center_expr.sum(axis=0)).flatten()
        sum_sq_center = np.array(center_expr.power(2).sum(axis=0)).flatten()

        # ==================== Cross-covariance computation ====================
        # Term 1: (1/(n_c*n_nn)) Σ_i Σ_j x_i*y_j
        sum_non_neighbor = np.array(non_neighbor_expr.sum(axis=0)).flatten()
        term1 = np.outer(sum_center, sum_non_neighbor) / (n_center * n_non_neighbor)

        # Term 2: -(1/(n_c*n_nn)) Σ_i Σ_j x_i*μ^{ct_j}
        weighted_nn_mean = non_neighbor_type_means_matrix @ type_counts
        term2 = np.outer(sum_center, weighted_nn_mean) / (n_center * n_non_neighbor)

        # Term 3: -(μ_X/(n_c*n_nn)) Σ_i Σ_j y_j
        term3 = np.outer(center_mean, sum_non_neighbor) / n_non_neighbor

        # Term 4: (μ_X/(n_c*n_nn)) Σ_i Σ_j μ^{ct_j}
        term4 = np.outer(center_mean, weighted_nn_mean) / n_non_neighbor

        cross_cov = term1 - term2 - term3 + term4

        # ==================== Variance computation ====================
        # Var(X - μ_X)
        var_center = (sum_sq_center / n_center
                     - 2 * center_mean * sum_center / n_center
                     + center_mean**2)

        # Var(Y - μ^{ct_j})
        sum_sq_non_neighbor = np.array(non_neighbor_expr.power(2).sum(axis=0)).flatten()

        type_indicator = csr_matrix((np.ones(n_non_neighbor), (np.arange(n_non_neighbor), type_indices)),
                                   shape=(n_non_neighbor, n_types))
        sum_non_neighbor_by_type = non_neighbor_expr.T @ type_indicator
        if sparse.issparse(sum_non_neighbor_by_type):
            sum_non_neighbor_by_type = np.asarray(sum_non_neighbor_by_type.todense())

        term_y_mean = (sum_non_neighbor_by_type * non_neighbor_type_means_matrix).sum(axis=1) / n_non_neighbor
        term_mean_sq = ((non_neighbor_type_means_matrix**2) @ type_counts) / n_non_neighbor

        var_non_neighbor = sum_sq_non_neighbor / n_non_neighbor - 2 * term_y_mean + term_mean_sq

        std_center = np.sqrt(np.maximum(var_center, 0))
        std_non_neighbor = np.sqrt(np.maximum(var_non_neighbor, 0))
    else:
        # Dense matrix operations
        non_neighbor_type_means_matrix = np.array([cell_type_means[ct] for ct in non_neighbor_cell_types])

        # Center the data
        center_expr_shifted = center_expr - center_mean[np.newaxis, :]
        non_neighbor_expr_shifted = non_neighbor_expr - non_neighbor_type_means_matrix

        # For all-to-all pairs: sum over all combinations
        sum_center_shifted = center_expr_shifted.sum(axis=0)
        sum_non_neighbor_shifted = non_neighbor_expr_shifted.sum(axis=0)

        cross_cov = np.outer(sum_center_shifted, sum_non_neighbor_shifted) / (n_center * n_non_neighbor)

        # Variances
        var_center = (center_expr_shifted**2).sum(axis=0) / n_center
        var_non_neighbor = (non_neighbor_expr_shifted**2).sum(axis=0) / n_non_neighbor

        std_center = np.sqrt(np.maximum(var_center, 0))
        std_non_neighbor = np.sqrt(np.maximum(var_non_neighbor, 0))

    std_outer = np.outer(std_center, std_non_neighbor)
    std_outer[std_outer == 0] = 1e-10

    corr_matrix = cross_cov / std_outer

    # Effective sample size
    n_eff = min(n_center, n_non_neighbor)

    return corr_matrix, n_eff


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


def fisher_z_transform(r):
    """Fisher Z transformation for correlation coefficient"""
    r_clipped = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))

def fisher_z_test(r1, n1, r2, n2):
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


def test_score_difference(result_A: pd.DataFrame,
                         result_B: pd.DataFrame,
                         score_col: str = 'combined_score',
                         significance_col: str = 'if_significant',
                         gene_center_col: str = 'gene_center',
                         gene_motif_col: str = 'gene_motif',
                         percentile_threshold: float = 95.0) -> pd.DataFrame:
    """
    Test whether gene-pairs have significantly different correlation scores between two conditions.

    This function compares the combined_score (or another score column) between two groups of
    gene-pairs (e.g., from different motifs or conditions). It:
    1. Identifies overlapping gene-pairs present in both groups
    2. Filters for pairs that are significant in at least one group
    3. Computes score differences (scores_A - scores_B) for all overlapping pairs
    4. For each pair, computes its percentile rank in the score difference distribution
    5. Identifies outlier pairs (percentile > 95 or < 5) as significantly different

    Algorithm:
    ----------
    For overlapping gene-pairs X:
        diff_X = score_X_A - score_X_B

    For the distribution of all differences diff_all:
        percentile_X = percentileofscore(diff_all, diff_X)

    If percentile_X > 95 or < 5:
        → pair X is an outlier (significantly higher/lower in group A)

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

    Examples
    --------
    >>> # Compare gene-pairs between two motifs
    >>> result_motif1, _ = sq.compute_gene_gene_correlation_by_type(
    ...     ct='Gut tube', motif=['Endothelium', 'Haematoendothelial progenitors'],
    ...     max_dist=5)
    >>> result_motif2, _ = sq.compute_gene_gene_correlation_by_type(
    ...     ct='Gut tube', motif=['Splanchnic mesoderm'],
    ...     max_dist=5)
    >>>
    >>> # Test score differences
    >>> diff_results = test_score_difference(result_motif1, result_motif2)
    >>>
    >>> # Get significantly different pairs
    >>> significant_diff = diff_results[diff_results['is_outlier']]
    >>> print(f"Found {len(significant_diff)} significantly different pairs")
    >>>
    >>> # Get pairs higher in motif1
    >>> higher_in_A = diff_results[diff_results['outlier_direction'] == 'higher_in_A']
    """
    from scipy.stats import percentileofscore

    # Validate inputs
    required_cols = [gene_center_col, gene_motif_col, score_col, significance_col]
    for col in required_cols:
        if col not in result_A.columns:
            raise ValueError(f"Column '{col}' not found in result_A")
        if col not in result_B.columns:
            raise ValueError(f"Column '{col}' not found in result_B")

    # Create pair identifiers
    result_A = result_A.copy()
    result_B = result_B.copy()
    result_A['pair_id'] = result_A[gene_center_col] + '|' + result_A[gene_motif_col]
    result_B['pair_id'] = result_B[gene_center_col] + '|' + result_B[gene_motif_col]

    # Find pairs that are significant in at least one group
    sig_pairs_A = set(result_A[result_A[significance_col]]['pair_id'])
    sig_pairs_B = set(result_B[result_B[significance_col]]['pair_id'])
    at_least_one_sig = sig_pairs_A | sig_pairs_B

    print(f"Total pairs in group A: {len(result_A)}")
    print(f"Total pairs in group B: {len(result_B)}")
    print(f"Significant pairs in A: {len(sig_pairs_A)}")
    print(f"Significant pairs in B: {len(sig_pairs_B)}")
    print(f"Pairs significant in at least one group: {len(at_least_one_sig)}")

    # Find overlapping pairs (present in both groups)
    pairs_A = set(result_A['pair_id'])
    pairs_B = set(result_B['pair_id'])
    overlapping_pairs = pairs_A & pairs_B

    print(f"Overlapping pairs: {len(overlapping_pairs)}")

    # Filter for overlapping pairs that are significant in at least one group
    pairs_to_test = overlapping_pairs & at_least_one_sig

    if len(pairs_to_test) == 0:
        raise ValueError("No overlapping gene-pairs found that are significant in at least one group")

    print(f"Pairs to test (overlapping AND significant in at least one): {len(pairs_to_test)}")

    # Extract scores for these pairs
    result_A_filtered = result_A[result_A['pair_id'].isin(pairs_to_test)].copy()
    result_B_filtered = result_B[result_B['pair_id'].isin(pairs_to_test)].copy()

    # Merge to ensure alignment
    merged = result_A_filtered.merge(
        result_B_filtered,
        on='pair_id',
        suffixes=('_A', '_B')
    )

    # Calculate score differences
    score_col_A = score_col + '_A'
    score_col_B = score_col + '_B'
    merged['score_diff'] = merged[score_col_A] - merged[score_col_B]

    # Compute percentile for each pair
    diff_all = merged['score_diff'].values
    merged['percentile'] = merged['score_diff'].apply(
        lambda x: percentileofscore(diff_all, x, kind='rank')
    )

    # Identify outliers
    lower_threshold = 100 - percentile_threshold
    merged['is_outlier'] = (merged['percentile'] > percentile_threshold) | \
                          (merged['percentile'] < lower_threshold)

    # Classify outlier direction
    merged['outlier_direction'] = 'not_outlier'
    merged.loc[merged['percentile'] > percentile_threshold, 'outlier_direction'] = 'higher_in_A'
    merged.loc[merged['percentile'] < lower_threshold, 'outlier_direction'] = 'lower_in_A'

    # Prepare output columns
    output_cols = [
        gene_center_col + '_A',
        gene_motif_col + '_A',
        score_col_A,
        score_col_B,
        'score_diff',
        'percentile',
        'is_outlier',
        significance_col + '_A',
        significance_col + '_B',
        'outlier_direction'
    ]

    result = merged[output_cols].copy()
    result.columns = [
        'gene_center',
        'gene_motif',
        'score_A',
        'score_B',
        'score_diff',
        'percentile',
        'is_outlier',
        'significant_in_A',
        'significant_in_B',
        'outlier_direction'
    ]

    # Sort by absolute score difference (descending)
    result['abs_score_diff'] = np.abs(result['score_diff'])
    result = result.sort_values('abs_score_diff', ascending=False).reset_index(drop=True)
    result = result.drop('abs_score_diff', axis=1)

    # Print summary
    n_outliers = result['is_outlier'].sum()
    n_higher_A = (result['outlier_direction'] == 'higher_in_A').sum()
    n_lower_A = (result['outlier_direction'] == 'lower_in_A').sum()

    print(f"\n{'='*60}")
    print(f"Score Difference Test Results")
    print(f"{'='*60}")
    print(f"Total pairs tested: {len(result)}")
    print(f"Outlier pairs (percentile > {percentile_threshold} or < {100-percentile_threshold}): {n_outliers}")
    print(f"  - Higher in A: {n_higher_A}")
    print(f"  - Lower in A: {n_lower_A}")
    print(f"\nScore difference range: [{result['score_diff'].min():.3f}, {result['score_diff'].max():.3f}]")
    print(f"Mean score difference: {result['score_diff'].mean():.3f}")
    print(f"Std score difference: {result['score_diff'].std():.3f}")

    return result
