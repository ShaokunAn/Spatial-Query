
from token import OP
from typing import List, Union, Optional, Literal, Dict

import numpy as np
import pandas as pd
from scipy import sparse

from scipy.sparse import csr_matrix
from statsmodels.stats.multitest import multipletests
from scipy import stats as scipy_stats

from scipy.stats import percentileofscore

from time import time

from . import spatial_utils


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

def compute_cross_correlation_paired(expr_genes,
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

def compute_cross_correlation_all_to_all(expr_genes,
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
        # To avoid overflow, normalize before outer product

        # Compute normalized sums (mean × count)
        sum_non_neighbor = np.array(non_neighbor_expr.sum(axis=0)).flatten()
        mean_center = sum_center / n_center
        mean_non_neighbor = sum_non_neighbor / n_non_neighbor

        # Term 1: (1/(n_c*n_nn)) Σ_i Σ_j x_i*y_j
        # = np.outer(sum_center / n_center, sum_non_neighbor / n_non_neighbor)
        term1 = np.outer(mean_center, mean_non_neighbor)

        # Term 2: -(1/(n_c*n_nn)) Σ_i Σ_j x_i*μ^{ct_j}
        weighted_nn_mean = non_neighbor_type_means_matrix @ type_counts
        mean_weighted_nn = weighted_nn_mean / n_non_neighbor
        term2 = np.outer(mean_center, mean_weighted_nn)

        # Term 3: -(μ_X/(n_c*n_nn)) Σ_i Σ_j y_j
        # = -np.outer(center_mean, sum_non_neighbor / n_non_neighbor)
        term3 = np.outer(center_mean, mean_non_neighbor)

        # Term 4: (μ_X/(n_c*n_nn)) Σ_i Σ_j μ^{ct_j}
        # = np.outer(center_mean, weighted_nn_mean / n_non_neighbor)
        term4 = np.outer(center_mean, mean_weighted_nn)

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
        # To avoid overflow, normalize before outer product
        sum_center_shifted = center_expr_shifted.sum(axis=0)
        sum_non_neighbor_shifted = non_neighbor_expr_shifted.sum(axis=0)

        mean_center_shifted = sum_center_shifted / n_center
        mean_non_neighbor_shifted = sum_non_neighbor_shifted / n_non_neighbor

        cross_cov = np.outer(mean_center_shifted, mean_non_neighbor_shifted)

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
        # To avoid overflow, normalize sums before outer product
        sum_center = np.array(center_expr.sum(axis=0)).flatten()
        sum_neighbor = np.array(neighbor_expr.sum(axis=0)).flatten()

        # Normalize by counts to avoid overflow in outer product
        mean_center = sum_center / n_centers
        mean_neighbor = sum_neighbor / n_neighbors

        # Term 1: Σ_i Σ_j x_i * y_j = n_centers * n_neighbors * mean_center * mean_neighbor
        term1 = n_pairs * np.outer(mean_center, mean_neighbor)

        # Term 2: -Σ_i Σ_j x_i * μ^{ct_j} = n_centers * (Σ_i x_i) * (Σ_j μ^{ct_j}) / n_neighbors
        weighted_neighbor_mean = neighbor_type_means_matrix @ type_counts
        mean_weighted_neighbor = weighted_neighbor_mean / n_neighbors
        term2 = n_centers * np.outer(mean_center, mean_weighted_neighbor) * n_neighbors

        # Term 3: -Σ_i Σ_j μ_X * y_j = n_centers * μ_X * (Σ_j y_j)
        term3 = n_centers * np.outer(center_mean, mean_neighbor) * n_neighbors

        # Term 4: Σ_i Σ_j μ_X * μ^{ct_j} = n_centers * n_neighbors * μ_X * mean(μ^{ct_j})
        term4 = n_pairs * np.outer(center_mean, mean_weighted_neighbor)

        cov_sum = term1 - term2 - term3 + term4

        # ==================== Sum of squares computation ====================
        # Center: Σ_i Σ_j (x_i - μ_X)^2 = n_neighbors * Σ_i (x_i - μ_X)^2
        sum_sq_center = np.array(center_expr.power(2).sum(axis=0)).flatten()
        # Rewrite to avoid large intermediate values
        center_variance_sum = sum_sq_center - 2 * center_mean * sum_center + n_centers * center_mean**2
        center_ss = n_neighbors * center_variance_sum

        # Neighbor: Σ_i Σ_j (y_j - μ^{ct_j})^2 = n_centers * Σ_j (y_j - μ^{ct_j})^2
        sum_sq_neighbor = np.array(neighbor_expr.power(2).sum(axis=0)).flatten()

        type_indicator = csr_matrix((np.ones(n_neighbors), (np.arange(n_neighbors), type_indices)),
                                   shape=(n_neighbors, n_types))

        sum_neighbor_by_type = neighbor_expr.T @ type_indicator
        if sparse.issparse(sum_neighbor_by_type):
            sum_neighbor_by_type = np.asarray(sum_neighbor_by_type.todense())

        term_y_mean = (sum_neighbor_by_type * neighbor_type_means_matrix).sum(axis=1)
        term_mean_sq = (neighbor_type_means_matrix**2) @ type_counts

        neighbor_variance_sum = sum_sq_neighbor - 2 * term_y_mean + term_mean_sq
        neighbor_ss = n_centers * neighbor_variance_sum

    else:
        # Dense matrix operations
        center_shifted = center_expr - center_mean[np.newaxis, :]

        neighbor_type_means_matrix = np.array([cell_type_means[ct] for ct in neighbor_cell_types])
        neighbor_shifted = neighbor_expr - neighbor_type_means_matrix

        # Compute covariance sum: Σ_i Σ_j x'_i * y'_j = (Σ_i x'_i) @ (Σ_j y'_j).T
        # To avoid overflow, normalize before outer product
        sum_center_shifted = center_shifted.sum(axis=0)
        sum_neighbor_shifted = neighbor_shifted.sum(axis=0)

        mean_center_shifted = sum_center_shifted / n_centers
        mean_neighbor_shifted = sum_neighbor_shifted / n_neighbors

        cov_sum = n_pairs * np.outer(mean_center_shifted, mean_neighbor_shifted)

        # Compute sum of squares
        center_ss = n_neighbors * (center_shifted**2).sum(axis=0)
        neighbor_ss = n_centers * (neighbor_shifted**2).sum(axis=0)

    return cov_sum, center_ss, neighbor_ss, n_pairs, n_eff


def compute_gene_gene_correlation_adata(sq_obj,
                                        ct: str,
                                        motif: Union[str, List[str]],
                                        genes: Optional[Union[str, List[str]]] = None,
                                        max_dist: Optional[float] = None,
                                        k: Optional[int] = None,
                                        min_size: int = 0,
                                        min_nonzero: int = 10,
                                        alpha: Optional[float] = None
                                        ) -> pd.DataFrame:
    """
    Compute gene-gene cross correlation between neighbor and non-neighbor motif cells. Only considers inter-cell-type interactions. 
    After finding neighbors using the full motif, removes all cells of the center cell type from both neighbor and
    non-neighbor groups. For Pearson correlation, uses shifted correlation (subtract cell type mean) to enable
    comparison across different niches/motifs.

    This function calculates cross correlation between gene expression in:
    1. Motif cells that are neighbors of center cell type (excluding center type cells in neighbor group)
    2. Motif cells that are NOT neighbors of center cell type (excluding center type cells)
    3. Neighboring cells of center cell type without nearby motif

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
    alpha:
        Significance threshold

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
            - combined_score: combined significance score
            - abs_combined_score: absolute value of combined score
            - if_significant: whether both tests pass FDR threshold

    cell_groups : dict
        Dictionary containing cell pairing information for correlations:
            - 'center_neighbor_motif_pair': array of shape (n_pairs, 2) containing
                center-neighbor pairs for Correlation 1 (center with motif vs neighboring motif).
                Each row is [center_cell_idx, neighbor_cell_idx].
            - 'non-neighbor_motif_cells': array of cell indices for distant motif cells
                used in Correlation 2 (center with motif vs distant motif).
                Correlation 2 uses all combinations of center cells (from corr1) × these cells.
            - 'non_motif_center_neighbor_pair': array of shape (n_pairs, 2) containing
                center-neighbor pairs for Correlation 3 (center without motif vs neighbors).
                Each row is [center_cell_idx, neighbor_cell_idx]. Empty if insufficient pairs.

        Note: Individual cell IDs can be extracted from pairs using np.unique() like:
            - center_cells = np.unique(center_neighbor_motif_pair[:, 0])
            - neighbor_cells = np.unique(center_neighbor_motif_pair[:, 1])
    """
    
    motif = motif if isinstance(motif, list) else [motif]
    if alpha is None:
        alpha = 0.05

    # Get neighbor and non-neighbor cell IDs plus paired mappings (using original motif)
    neighbor_result = spatial_utils.get_motif_neighbor_cells(sq_obj=sq_obj, ct=ct, motif=motif, max_dist=max_dist, k=k, min_size=min_size)

    # Extract paired data and derive cell IDs from pairs
    center_neighbor_pairs = neighbor_result['center_neighbor_pairs']
    ct_in_motif = neighbor_result['ct_in_motif']

    # Extract unique center and neighbor cells from pairs
    # Note: if ct is in motif, center-type neighbors are already placed in first column as centers
    center_cells = np.unique(center_neighbor_pairs[:, 0])  # unique center cells
    neighbor_cells = np.unique(center_neighbor_pairs[:, 1])  # neighboring motif cells without center type

    # Get non-neighbor motif cells (set difference from all motif cells)
    motif_mask = np.isin(np.array(sq_obj.labels), motif)
    all_motif_cells = np.where(motif_mask)[0]
    non_neighbor_cells = np.setdiff1d(all_motif_cells, neighbor_cells)

    # Remove cells of center type from non-neighbor groups to focus on inter-cell-type interactions
    if ct_in_motif:
        center_cell_mask_non = sq_obj.labels[non_neighbor_cells] == ct
        non_neighbor_cells = non_neighbor_cells[~center_cell_mask_non]
        print(f'Focus on inter-cell-type interactions: Remove center type cells from non-neighbor groups.')

    if len(non_neighbor_cells) < 10:
        raise ValueError(f"Not enough non-neighbor cells ({len(non_neighbor_cells)}) for correlation analysis. Need at least 5 cells.")
    

    # Get gene list
    if genes is None:
        genes = sq_obj.genes
    elif isinstance(genes, str):
        genes = [genes]

    # Filter genes that exist
    valid_genes = [g for g in genes if g in sq_obj.genes]
    if len(valid_genes) == 0:
        raise ValueError("No valid genes found in the dataset.")
    expr_genes = sq_obj.adata[:, valid_genes].X

    # Check if sparse
    is_sparse = sparse.issparse(expr_genes)

    # Filter genes by non-zero expression (work with sparse matrix)
    if is_sparse:
        nonzero_all = np.array((expr_genes > 0).sum(axis=0)).flatten()
    else:
        nonzero_all = (expr_genes > 0).sum(axis=0)

    valid_gene_mask = nonzero_all >= min_nonzero

    if valid_gene_mask.sum() == 0:
        raise ValueError(f"No genes passed the min_nonzero={min_nonzero} filter for cells of interest.")

    # Apply gene filter
    filtered_genes = [valid_genes[i] for i in range(len(valid_genes)) if valid_gene_mask[i]]
    expr_genes= expr_genes[:, valid_gene_mask]

    print(f"After filtering (min_nonzero={min_nonzero}): {len(filtered_genes)} genes")

    # Compute cell type means for ALL cell types (keep sparse)
    # We need cell-type-specific means for proper centering
    start_time = time()

    # Get all unique cell types in the dataset
    all_cell_types = np.unique(sq_obj.labels)
    cell_type_means = {}  # Dictionary to store mean expression for each cell type

    for cell_type in all_cell_types:
        ct_mask = sq_obj.labels == cell_type
        ct_cells = np.where(ct_mask)[0]
        if len(ct_cells) > 0:
            ct_expr = expr_genes[ct_cells, :]
            if is_sparse:
                cell_type_means[cell_type] = np.array(ct_expr.mean(axis=0)).flatten()
            else:
                cell_type_means[cell_type] = ct_expr.mean(axis=0)
    
    center_mean = cell_type_means[ct]


    # ==================================================================================
    # Correlation 1: Center with motif vs Neighboring motif (PAIRED DATA)
    # ==================================================================================
    print("\n" + "="*60)
    print("Computing Correlation 1: Center with motif vs Neighboring motif (paired)")
    print("="*60)

    # Convert pairs to arrays
    pair_centers = center_neighbor_pairs[:, 0]
    pair_neighbors = center_neighbor_pairs[:, 1]

    # Get cell types for each neighbor in pairs
    neighbor_cell_types = sq_obj.labels[pair_neighbors]

    # Compute correlation using helper function
    n_genes = len(filtered_genes)
    n_pairs = len(pair_centers)

    corr_matrix_neighbor, n_eff_neighbor = compute_cross_correlation_paired(
        expr_genes=expr_genes,
        pair_centers=pair_centers,
        pair_neighbors=pair_neighbors,
        center_mean=center_mean,
        cell_type_means=cell_type_means,
        neighbor_cell_types=neighbor_cell_types,
        is_sparse=is_sparse
    )

    print(f"Number of pairs: {n_pairs}")
    print(f"Unique center cells: {len(np.unique(pair_centers))}")
    print(f"Unique neighbor cells: {len(np.unique(pair_neighbors))}")
    print(f"Effective sample size: {n_eff_neighbor}")

    end_time = time()
    print(f"Time for computing correlation-1 matrix: {end_time - start_time:.4f} seconds")
    # ==================================================================================
    # Correlation 2: Center with motif vs Distant motif (ALL PAIRS)
    # ==================================================================================
    print("\n" + "="*60)
    print("Computing Correlation 2: Center with motif vs Distant motif (all pairs)")
    print("="*60)
    start_time = time()

    # Get cell types for non-neighbor cells
    non_neighbor_cell_types = sq_obj.labels[non_neighbor_cells]

    n_center = len(center_cells)
    n_non_neighbor = len(non_neighbor_cells)

    corr_matrix_non_neighbor, n_eff_non_neighbor = compute_cross_correlation_all_to_all(
        expr_genes=expr_genes,
        center_cells=center_cells,
        non_neighbor_cells=non_neighbor_cells,
        center_mean=center_mean,
        cell_type_means=cell_type_means,
        non_neighbor_cell_types=non_neighbor_cell_types,
        is_sparse=is_sparse
    )

    print(f"Unique center cells: {n_center}")
    print(f"Unique distant cells: {n_non_neighbor}")
    print(f"Effective sample size: {n_eff_non_neighbor}")

    end_time = time()
    print(f'Time for computing correlations-2 matrix: {end_time-start_time:.2f} seconds')

    # ==================================================================================
    # Correlation 3: Center without motif vs Neighbors (PAIRED DATA)
    # ==================================================================================
    print("\n" + "="*60)
    print("Computing Correlation 3: Center without motif vs Neighbors (paired)")
    print("="*60)

    # Get neighbors for centers WITHOUT motif by excluding centers with motif
    start_time = time()
    no_motif_result = spatial_utils.get_all_neighbor_cells(sq_obj=sq_obj,
        ct=ct,
        max_dist=max_dist,
        k=k,
        min_size=min_size,
        exclude_centers=center_cells,
        exclude_neighbors=neighbor_cells,
    )

    center_no_motif_pairs = no_motif_result['center_neighbor_pairs']

    if len(center_no_motif_pairs) < 10:
        print(f"Not enough pairs ({len(center_no_motif_pairs)}) for centers without motif. Skipping.")
        corr_matrix_no_motif = None
        n_eff_no_motif = 0
    else:
        # Extract paired data
        pair_centers_no_motif = center_no_motif_pairs[:, 0]
        pair_neighbors_no_motif = center_no_motif_pairs[:, 1]

        # Get cell types for each neighbor in pairs
        neighbor_no_motif_cell_types = sq_obj.labels[pair_neighbors_no_motif]

        corr_matrix_no_motif, n_eff_no_motif = compute_cross_correlation_paired(
            expr_genes=expr_genes,
            pair_centers=pair_centers_no_motif,
            pair_neighbors=pair_neighbors_no_motif,
            center_mean=center_mean,
            cell_type_means=cell_type_means,
            neighbor_cell_types=neighbor_no_motif_cell_types,
            is_sparse=is_sparse
        )
        
        print(f"Number of pairs: {len(center_no_motif_pairs)}")
        print(f"Unique center cells: {len(np.unique(pair_centers_no_motif))}")
        print(f"Unique neighbor cells: {len(np.unique(pair_neighbors_no_motif))}")
        print(f"Effective sample size: {n_eff_no_motif}")

    end_time = time()
    print(f"time of computing correlation-3 matrix is {end_time - start_time:.2f} seconds.")

    # Prepare results - combine all three correlations and perform statistical tests
    print("\n" + "="*60)
    print("Performing Fisher Z-tests and preparing results")
    print("="*60)

    start_time = time()

    # Test 1: Corr1 vs Corr2 (neighbor vs non-neighbor) - vectorized
    _, p_value_test1 = fisher_z_test(
        corr_matrix_neighbor, n_eff_neighbor,
        corr_matrix_non_neighbor, n_eff_non_neighbor
    )
    delta_corr_test1 = corr_matrix_neighbor - corr_matrix_non_neighbor

    # Test 2: Corr1 vs Corr3 (neighbor vs no_motif) - vectorized if available
    if corr_matrix_no_motif is not None:
        _, p_value_test2 = fisher_z_test(
            corr_matrix_neighbor, n_eff_neighbor,
            corr_matrix_no_motif, n_eff_no_motif
        )
        delta_corr_test2 = corr_matrix_neighbor - corr_matrix_no_motif

        # Combined score (vectorized)
        combined_score = (0.3 * delta_corr_test1 * (-np.log10(p_value_test1 + 1e-300)) +
                        0.7 * delta_corr_test2 * (-np.log10(p_value_test2 + 1e-300)))
    else:
        p_value_test2 = None
        delta_corr_test2 = None
        combined_score = None

    # Create meshgrid for gene pairs (vectorized)
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

    # Apply FDR correction accounting for ALL tests performed
    # Strategy: Pool all p-values from both tests, apply FDR jointly
    print(f"Total gene pairs: {len(results_df)}")

    if corr_matrix_no_motif is not None:
        # Step 1: Filter by direction consistency first
        # Both delta_corr should have the same sign (both positive or both negative)
        same_direction = np.sign(results_df['delta_corr_test1']) == np.sign(results_df['delta_corr_test2'])
        print(f"Gene pairs with consistent covarying direction: {same_direction.sum()}")

        if same_direction.sum() > 0:
            # Step 2: Pool all p-values from both tests for FDR correction
            # This accounts for the fact that we perform 2 �� n_gene_pairs tests
            p_values_test1 = results_df.loc[same_direction, 'p_value_test1'].values
            p_values_test2 = results_df.loc[same_direction, 'p_value_test2'].values

            assert len(p_values_test1) == len(p_values_test2), "Inconsistent number of p-values!"
            # Concatenate all p-values: total = 2 × n_gene_pairs
            all_p_values = np.concatenate([p_values_test1, p_values_test2])
            n_consistent = same_direction.sum()

            print(f"Applying FDR correction to {len(all_p_values)} total tests (2 × {n_consistent} gene pairs)")

            # Apply FDR correction to ALL pooled p-values
            reject_all, q_values_all, _, _ = multipletests(
                all_p_values,
                alpha=alpha,
                method='fdr_bh'
            )

            # Split results back to test1 and test2
            q_values_test1 = q_values_all[:n_consistent]
            q_values_test2 = q_values_all[n_consistent:]
            reject_test1 = reject_all[:n_consistent]
            reject_test2 = reject_all[n_consistent:]

            # Assign q-values and rejection status
            results_df.loc[same_direction, 'q_value_test1'] = q_values_test1
            results_df.loc[same_direction, 'q_value_test2'] = q_values_test2
            results_df.loc[same_direction, 'reject_test1_fdr'] = reject_test1
            results_df.loc[same_direction, 'reject_test2_fdr'] = reject_test2

            # For inconsistent direction, set q-values to NaN
            # 只保留same_direction的pair
            results_df = results_df[same_direction]

            # Count gene pairs passing both FDR thresholds
            mask_both_fdr = reject_test1 & reject_test2
            n_both_fdr = mask_both_fdr.sum()

            print(f"\nFDR correction results (joint across both tests):")
            print(f"  - Test1 FDR significant (q < {alpha}): {reject_test1.sum()}")
            print(f"  - Test2 FDR significant (q < {alpha}): {reject_test2.sum()}")
            print(f"  - Both tests FDR significant: {n_both_fdr}")
        else:
            results_df['q_value_test1'] = np.nan
            results_df['q_value_test2'] = np.nan
            results_df['reject_test1_fdr'] = False
            results_df['reject_test2_fdr'] = False
            print("No gene pairs with consistent direction found.")
    else:
        # Only test1 available - apply FDR to all test1 p-values
        print("Note: Test2 not available (no centers without motif)")
        p_values_test1_all = results_df['p_value_test1'].values
        reject_test1, q_values_test1, _, _ = multipletests(
            p_values_test1_all,
            alpha=alpha,
            method='fdr_bh'
        )
        results_df['q_value_test1'] = q_values_test1
        results_df['reject_test1_fdr'] = reject_test1
        results_df['q_value_test2'] = np.nan
        results_df['reject_test2_fdr'] = False

        print(f"FDR correction applied to all {len(results_df)} gene pairs:")
        print(f"  - Test1 FDR significant (q < {alpha}): {reject_test1.sum()}")

    # Sort by absolute value of combined score (descending) to capture both positive and negative co-varying
    if corr_matrix_no_motif is not None:
        results_df['abs_combined_score'] = np.abs(results_df['combined_score'])
        results_df = results_df.sort_values('abs_combined_score', ascending=False, na_position='last').reset_index(drop=True)
        results_df['if_significant'] = results_df['reject_test1_fdr'] & results_df['reject_test2_fdr']
    else:
        # Sort by absolute value of test1 score if test2 not available
        results_df['test1_score'] = results_df['delta_corr_test1'] * (-np.log10(results_df['p_value_test1'] + 1e-300))
        results_df['combined_score'] = results_df['test1_score']
        results_df['abs_combined_score'] = np.abs(results_df['combined_score'])
        results_df = results_df.sort_values('abs_combined_score', ascending=False, na_position='last')
        results_df['if_significant'] = results_df['reject_test1_fdr'] & results_df['reject_test2_fdr']


    end_time = time()
    print(f"Time for testing results: {end_time - start_time:.2f} seconds\n")

    # Prepare cell groups dictionary with cell pairs for each correlation
    cell_groups = {
        'center_neighbor_motif_pair': center_neighbor_pairs,  # Shape: (n_pairs, 2), columns: [center_idx, neighbor_idx]
        'non-neighbor_motif_cells': non_neighbor_cells,
        'non_motif_center_neighbor_pair': center_no_motif_pairs if corr_matrix_no_motif is not None else np.array([]).reshape(0, 2),
    }

    return results_df, cell_groups

def compute_gene_gene_correlation_binary(sq_obj,
                                         ct: str,
                                         motif: Union[str, List[str]],
                                         genes: Optional[Union[str, List[str]]] = None,
                                         max_dist: Optional[float] = None,
                                         k: Optional[int] = None,
                                         min_size: int = 0,
                                         min_nonzero: int = 10,
                                         alpha: Optional[float] = None
                                         ) -> pd.DataFrame:
    """
    Compute gene-gene correlation using deviation from global binary mean for binary expression data.

    Instead of computing phi coefficient directly on binary variables, this method:
    1. Computes global binary expression mean for each gene in each cell type
    2. For each cell, computes deviation = binary_value - global_mean
    3. Computes Pearson correlation between deviation vectors across gene pairs
    4. Compares correlations using Fisher's Z test

    The analysis structure:
    1. Correlation 1: Center with motif vs Neighboring motif (paired)
    2. Correlation 2: Center with motif vs Distant motif (all-to-all)
    3. Correlation 3: Center without motif vs Neighbors (paired)

    Parameters
    ----------
    ct : str
        Cell type as the center cells.
    motif : Union[str, List[str]]
        Motif (names of cell types) to be analyzed.
    genes : Optional[Union[str, List[str]]], default=None
        List of genes to analyze. If None, all genes in the index will be used.
    max_dist : Optional[float], default=None
        Maximum distance for considering a cell as a neighbor. Use either max_dist or k.
    k : Optional[int], default=None
        Number of nearest neighbors. Use either max_dist or k.
    min_size : int, default=0
        Minimum neighborhood size for each center cell.
    min_nonzero : int, default=10
        Minimum number of non-zero expression values required for a gene to be included.
    alpha: 
        Significance threshold.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with deviation-based correlation results between neighbor and non-neighbor groups.
    cell_groups : dict
        Dictionary containing cell pairing information.
    """
    motif = motif if isinstance(motif, list) else [motif]
    if alpha is None:
        alpha = 0.1

    # Get neighbor and non-neighbor cell IDs using the same logic as compute_gene_gene_correlation
    neighbor_result = spatial_utils.get_motif_neighbor_cells(
        sq_obj=sq_obj, ct=ct, motif=motif, max_dist=max_dist, k=k, min_size=min_size
    )

    center_neighbor_pairs = neighbor_result['center_neighbor_pairs']
    ct_in_motif = neighbor_result['ct_in_motif']

    # Extract unique center and neighbor cells from pairs
    center_cells = np.unique(center_neighbor_pairs[:, 0])
    neighbor_cells = np.unique(center_neighbor_pairs[:, 1])

    # Get non-neighbor motif cells
    motif_mask = np.isin(sq_obj.labels, motif)
    all_motif_cells = np.where(motif_mask)[0]
    non_neighbor_cells = np.setdiff1d(all_motif_cells, neighbor_cells)

    # Remove center type cells from non-neighbor groups
    if ct_in_motif:
        center_cell_mask_non = sq_obj.labels[non_neighbor_cells] == ct
        non_neighbor_cells = non_neighbor_cells[~center_cell_mask_non]
        print(f'Focus on inter-cell-type interactions: Remove center type cells from non-neighbor groups.')

    if len(non_neighbor_cells) < 10:
        raise ValueError(f"Not enough non-neighbor cells ({len(non_neighbor_cells)}) for correlation analysis.")

    # Get gene list
    if genes is None:
        valid_genes = sq_obj.genes
    elif isinstance(genes, str):
        genes = [genes]
        valid_genes = [g for g in genes if g in sq_obj.index.scfindGenes]
    else:
        valid_genes = [g for g in genes if g in sq_obj.index.scfindGenes]

    if len(valid_genes) == 0:
        raise ValueError("No valid genes found in the scfind index.")

    print(f"Building binary expression matrix from scfind index for {len(valid_genes)} genes...")
    start_time = time()

    # Use efficient C++ method to get sparse matrix data directly
    # Returns: {'rows': np.array, 'cols': np.array, 'gene_names': list, 'n_cells': int}
    sparse_data = sq_obj.index.index.getBinarySparseMatrixData(valid_genes, sq_obj.dataset, min_nonzero)

    rows = sparse_data['rows']
    cols = sparse_data['cols']
    filtered_genes = sparse_data['gene_names']
    n_cells = sparse_data['n_cells']

    if len(filtered_genes) == 0:
        raise ValueError(f"No genes passed the min_nonzero={min_nonzero} filter.")

    # Create binary sparse matrix (cells × genes) directly from C++ output
    binary_expr = csr_matrix(
        (np.ones(len(rows), dtype=np.int16), (rows, cols)),
        shape=(n_cells, len(filtered_genes)),
    )
    print(f"Time for recovering binary matrix: {time() - start_time:.2f} seconds")

    # ==================================================================================
    # Compute global binary means for ALL cell types (similar to compute_gene_gene_correlation)
    # ==================================================================================
    print("\n" + "="*60)
    print("Computing global binary expression means for all cell types")
    print("="*60)
    start_time = time()

    # Get all unique cell types in the dataset
    all_cell_types = np.unique(sq_obj.labels)
    cell_type_means = {}  # Dictionary to store mean expression for each cell type

    for cell_type in all_cell_types:
        ct_mask = sq_obj.labels == cell_type
        ct_cells = np.where(ct_mask)[0]
        if len(ct_cells) > 0:
            ct_expr = binary_expr[ct_cells, :]
            cell_type_means[cell_type] = np.array(ct_expr.mean(axis=0)).flatten()

    center_mean = cell_type_means[ct]
    print(f"\nTime for computing global means: {time() - start_time:.2f} seconds")

    # ==================================================================================
    # Correlation 1: Center with motif vs Neighboring motif (PAIRED)
    # ==================================================================================
    print("\n" + "="*60)
    print("Computing Correlation 1: Center with motif vs Neighboring motif (paired)")
    print("="*60)
    start_time = time()

    pair_centers = center_neighbor_pairs[:, 0]
    pair_neighbors = center_neighbor_pairs[:, 1]

    # Get cell types for each neighbor in pairs
    neighbor_cell_types = sq_obj.labels[pair_neighbors]

    corr_matrix_neighbor, n_eff_neighbor = compute_cross_correlation_paired(
        expr_genes=binary_expr,
        pair_centers=pair_centers,
        pair_neighbors=pair_neighbors,
        center_mean=center_mean,
        cell_type_means=cell_type_means,
        neighbor_cell_types=neighbor_cell_types,
        is_sparse=True
    )

    print(f"Number of pairs: {len(pair_centers)}")
    print(f"Unique center cells: {len(np.unique(pair_centers))}")
    print(f"Unique neighbor cells: {len(np.unique(pair_neighbors))}")
    print(f"Effective sample size: {n_eff_neighbor}")
    print(f"Time for computing Corr-1: {time() - start_time:.4f} seconds")

    # ==================================================================================
    # Correlation 2: Center with motif vs Distant motif (ALL PAIRS)
    # ==================================================================================
    print("\n" + "="*60)
    print("Computing Correlation 2: Center with motif vs Distant motif (all pairs)")
    print("="*60)
    start_time = time()

    # Get cell types for non-neighbor cells
    non_neighbor_cell_types = sq_obj.labels[non_neighbor_cells]

    corr_matrix_non_neighbor, n_eff_non_neighbor = compute_cross_correlation_all_to_all(
        expr_genes=binary_expr,
        center_cells=center_cells,
        non_neighbor_cells=non_neighbor_cells,
        center_mean=center_mean,
        cell_type_means=cell_type_means,
        non_neighbor_cell_types=non_neighbor_cell_types,
        is_sparse=True
    )

    print(f"Unique center cells: {len(center_cells)}")
    print(f"Unique distant cells: {len(non_neighbor_cells)}")
    print(f"Effective sample size: {n_eff_non_neighbor}")
    print(f"Time for computing Corr-2: {time() - start_time:.2f} seconds")

    # ==================================================================================
    # Correlation 3: Center without motif vs Neighbors (PAIRED)
    # ==================================================================================
    print("\n" + "="*60)
    print("Computing Correlation 3: Center without motif vs Neighbors (paired)")
    print("="*60)
    start_time = time()

    no_motif_result = spatial_utils.get_all_neighbor_cells(
        sq_obj=sq_obj,
        ct=ct,
        max_dist=max_dist,
        k=k,
        min_size=min_size,
        exclude_centers=center_cells,
        exclude_neighbors=neighbor_cells,
    )

    center_no_motif_pairs = no_motif_result['center_neighbor_pairs']

    if len(center_no_motif_pairs) < 10:
        print(f"Not enough pairs ({len(center_no_motif_pairs)}) for centers without motif. Skipping.")
        corr_matrix_no_motif = None
        n_eff_no_motif = 0
    else:
        pair_centers_no_motif = center_no_motif_pairs[:, 0]
        pair_neighbors_no_motif = center_no_motif_pairs[:, 1]

        # Get cell types for each neighbor in no-motif pairs
        neighbor_cell_types_no_motif = sq_obj.labels[pair_neighbors_no_motif]

        corr_matrix_no_motif, n_eff_no_motif = compute_cross_correlation_paired(
            expr_genes=binary_expr,
            pair_centers=pair_centers_no_motif,
            pair_neighbors=pair_neighbors_no_motif,
            center_mean=center_mean,
            cell_type_means=cell_type_means,
            neighbor_cell_types=neighbor_cell_types_no_motif,
            is_sparse=True
        )

        print(f"Number of pairs: {len(center_no_motif_pairs)}")
        print(f"Unique center cells: {len(np.unique(pair_centers_no_motif))}")
        print(f"Unique neighbor cells: {len(np.unique(pair_neighbors_no_motif))}")
        print(f"Effective sample size: {n_eff_no_motif}")

    print(f"Time for computing Corr-3: {time() - start_time:.2f} seconds")

    # Prepare results using the same structure as compute_gene_gene_correlation
    print("\n" + "="*60)
    print("Performing Fisher's Z-tests for comparing correlations")
    print("="*60)
    start_time = time()

    # Test 1: Corr1 vs Corr2 (neighbor vs non-neighbor)
    _, p_value_test1 = fisher_z_test(
        corr_matrix_neighbor, n_eff_neighbor,
        corr_matrix_non_neighbor, n_eff_non_neighbor
    )
    delta_corr_test1 = corr_matrix_neighbor - corr_matrix_non_neighbor

    # Test 2: Corr1 vs Corr3 (neighbor vs no_motif)
    if corr_matrix_no_motif is not None:
        _, p_value_test2 = fisher_z_test(
            corr_matrix_neighbor, n_eff_neighbor,
            corr_matrix_no_motif, n_eff_no_motif
        )
        delta_corr_test2 = corr_matrix_neighbor - corr_matrix_no_motif

        # Combined score
        combined_score = (0.3 * delta_corr_test1 * (-np.log10(p_value_test1 + 1e-300)) +
                        0.7 * delta_corr_test2 * (-np.log10(p_value_test2 + 1e-300)))
    else:
        p_value_test2 = None
        delta_corr_test2 = None
        combined_score = None

    # Build results DataFrame
    n_genes = len(filtered_genes)
    gene_center_idx, gene_motif_idx = np.meshgrid(np.arange(n_genes), np.arange(n_genes), indexing='ij')

    results_df = pd.DataFrame({
        'gene_center': np.array(filtered_genes)[gene_center_idx.flatten()],
        'gene_motif': np.array(filtered_genes)[gene_motif_idx.flatten()],
        'corr_neighbor': corr_matrix_neighbor.flatten(),
        'corr_non_neighbor': corr_matrix_non_neighbor.flatten(),
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
        results_df['corr_center_no_motif'] = np.nan
        results_df['p_value_test2'] = np.nan
        results_df['delta_corr_test2'] = np.nan
        results_df['combined_score'] = np.nan

    # Apply FDR correction
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

            reject_all, q_values_all, _, _ = multipletests(all_p_values, alpha=alpha, method='fdr_bh')

            q_values_test1 = q_values_all[:n_consistent]
            q_values_test2 = q_values_all[n_consistent:]
            reject_test1 = reject_all[:n_consistent]
            reject_test2 = reject_all[n_consistent:]

            results_df.loc[same_direction, 'q_value_test1'] = q_values_test1
            results_df.loc[same_direction, 'q_value_test2'] = q_values_test2
            results_df.loc[same_direction, 'reject_test1_fdr'] = reject_test1
            results_df.loc[same_direction, 'reject_test2_fdr'] = reject_test2

            results_df = results_df[same_direction]

            mask_both_fdr = reject_test1 & reject_test2
            n_both_fdr = mask_both_fdr.sum()

            print(f"\nFDR correction results (joint across both tests):")
            print(f"  - Test1 FDR significant (q < {alpha}): {reject_test1.sum()}")
            print(f"  - Test2 FDR significant (q < {alpha}): {reject_test2.sum()}")
            print(f"  - Both tests FDR significant: {n_both_fdr}")
        else:
            results_df['q_value_test1'] = np.nan
            results_df['q_value_test2'] = np.nan
            results_df['reject_test1_fdr'] = False
            results_df['reject_test2_fdr'] = False
            print("No gene pairs with consistent direction found.")
    else:
        print("Note: Test2 not available (no centers without motif)")
        p_values_test1_all = results_df['p_value_test1'].values
        reject_test1, q_values_test1, _, _ = multipletests(p_values_test1_all, alpha=alpha, method='fdr_bh')
        results_df['q_value_test1'] = q_values_test1
        results_df['reject_test1_fdr'] = reject_test1
        results_df['q_value_test2'] = np.nan
        results_df['reject_test2_fdr'] = False

        print(f"FDR correction applied to all {len(results_df)} gene pairs:")
        print(f"  - Test1 FDR significant (q < {alpha}): {reject_test1.sum()}")

    # Sort by absolute value of combined score
    if corr_matrix_no_motif is not None:
        results_df['abs_combined_score'] = np.abs(results_df['combined_score'])
        results_df = results_df.sort_values('abs_combined_score', ascending=False, na_position='last').reset_index(drop=True)
        results_df['if_significant'] = results_df['reject_test1_fdr'] & results_df['reject_test2_fdr']
    else:
        results_df['test1_score'] = results_df['delta_corr_test1'] * (-np.log10(results_df['p_value_test1'] + 1e-300))
        results_df['combined_score'] = results_df['test1_score']
        results_df['abs_combined_score'] = np.abs(results_df['combined_score'])
        results_df = results_df.sort_values('abs_combined_score', ascending=False, na_position='last')
        results_df['if_significant'] = results_df['reject_test1_fdr'] & results_df['reject_test2_fdr']

    print(f"Time for testing results: {time() - start_time:.2f} seconds\n")

    # Prepare cell groups dictionary
    cell_groups = {
        'center_neighbor_motif_pair': center_neighbor_pairs,
        'non-neighbor_motif_cells': non_neighbor_cells,
        'non_motif_center_neighbor_pair': center_no_motif_pairs if corr_matrix_no_motif is not None else np.array([]).reshape(0, 2),
    }

    return results_df, cell_groups

def compute_gene_gene_correlation_by_type_adata(sq_obj,
                                                ct: str,
                                                motif: Union[str, List[str]],
                                                genes: Optional[Union[str, List[str]]] = None,
                                                max_dist: Optional[float] = None,
                                                k: Optional[int] = None,
                                                min_size: int = 0,
                                                min_nonzero: int = 10,
                                                alpha: Optional[float] = None
                                                ) -> pd.DataFrame:
    """
    Compute gene-gene cross correlation separately for each cell type in the motif.
    For each non-center cell type in the motif, compute:
    - Correlation 1: Center cells with motif vs neighboring motif cells of THIS TYPE
    - Correlation 2: Center cells with motif vs distant motif cells of THIS TYPE
    - Correlation 3: Center cells without motif vs neighbors (same for all types)

    Only analyzes motifs with >= 2 cell types besides the center type.

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
    alpha:
        Significance threshold

    Return
    ------
    results_df : DataFrame
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
            - if_significant: whether both tests pass FDR threshold
    """
    motif = motif if isinstance(motif, list) else [motif]

    # Get non-center cell types in motif
    non_center_types = [m for m in motif if m != ct]

    if alpha is None:
        alpha = 0.05

    if len(non_center_types) == 1:
        print(f"Only one non-center cell type in motif: {non_center_types}. Using compute_gene_gene_correlation method.")
        result, _ = compute_gene_gene_correlation_adata(
            sq_obj=sq_obj,
            ct=ct,
            motif=motif,
            genes=genes,
            max_dist=max_dist,
            k=k,
            min_size=min_size,
            min_nonzero=min_nonzero,
            alpha=alpha,
        )
        result['cell_type'] = non_center_types[0]
        return result
    elif len(non_center_types) == 0:
        raise ValueError("Error: Only center cell type in motif. Please ensure motif includes at least one non-center cell type.")

    print(f"Analyzing {len(non_center_types)} non-center cell types in motif: {non_center_types}")
    print("="*80)

    # Get neighbor and non-neighbor cell IDs using original motif
    neighbor_result = spatial_utils.get_motif_neighbor_cells(sq_obj=sq_obj, ct=ct, motif=motif, max_dist=max_dist, k=k, min_size=min_size)

    # Extract paired data
    center_neighbor_pairs = neighbor_result['center_neighbor_pairs']
    ct_in_motif = neighbor_result['ct_in_motif']

    # Extract unique center and neighbor cells from pairs
    center_cells = np.unique(center_neighbor_pairs[:, 0])
    neighbor_cells = np.unique(center_neighbor_pairs[:, 1])

    # Get all motif cells
    motif_mask = np.isin(np.array(sq_obj.labels), motif)
    all_motif_cells = np.where(motif_mask)[0]
    non_neighbor_cells = np.setdiff1d(all_motif_cells, neighbor_cells)

    # Remove center type cells from non-neighbor cells
    if ct_in_motif:
        center_cell_mask_non = sq_obj.labels[non_neighbor_cells] == ct
        non_neighbor_cells = non_neighbor_cells[~center_cell_mask_non]
        print(f'Removed {center_cell_mask_non.sum()} center type cells from non-neighbor group.')

    # Get gene list
    if genes is None:
        genes = sq_obj.genes
    elif isinstance(genes, str):
        genes = [genes]

    # Filter genes that exist
    valid_genes = [g for g in genes if g in sq_obj.genes]
    if len(valid_genes) == 0:
        raise ValueError("No valid genes found in the dataset.")
    expr_genes = sq_obj.adata[:, valid_genes].X

    # Check if sparse
    is_sparse = sparse.issparse(expr_genes)

    # Filter genes by non-zero expression
    if is_sparse:
        nonzero_all = np.array((expr_genes > 0).sum(axis=0)).flatten()
    else:
        nonzero_all = (expr_genes > 0).sum(axis=0)

    valid_gene_mask = nonzero_all >= min_nonzero

    if valid_gene_mask.sum() == 0:
        raise ValueError(f"No genes passed the min_nonzero={min_nonzero} filter.")

    # Apply gene filter
    filtered_genes = [valid_genes[i] for i in range(len(valid_genes)) if valid_gene_mask[i]]
    expr_genes = expr_genes[:, valid_gene_mask]

    print(f"After filtering (min_nonzero={min_nonzero}): {len(filtered_genes)} genes")

    # Compute cell type means for ALL cell types
    all_cell_types = np.unique(sq_obj.labels)
    cell_type_means = {}

    for cell_type in all_cell_types:
        ct_mask = sq_obj.labels == cell_type
        ct_cells = np.where(ct_mask)[0]
        if len(ct_cells) > 0:
            ct_expr = expr_genes[ct_cells, :]
            if is_sparse:
                cell_type_means[cell_type] = np.array(ct_expr.mean(axis=0)).flatten()
            else:
                cell_type_means[cell_type] = ct_expr.mean(axis=0)

    center_mean = cell_type_means[ct]
    n_genes = len(filtered_genes)

    # ==================================================================================
    # Correlation 3: Center without motif vs Neighbors (SAME FOR ALL TYPES)
    # ==================================================================================
    print("\n" + "="*80)
    print("Computing Correlation-3: Center without motif vs Neighbors")
    print("="*80)

    start_time = time()
    no_motif_result = spatial_utils.get_all_neighbor_cells(sq_obj=sq_obj,
        ct=ct,
        max_dist=max_dist,
        k=k,
        min_size=min_size,
        exclude_centers=center_cells,
        exclude_neighbors=neighbor_cells,
    )

    center_no_motif_pairs = no_motif_result['center_neighbor_pairs']

    if len(center_no_motif_pairs) < 10:
        print(f"Not enough pairs ({len(center_no_motif_pairs)}) for centers without motif. Skipping Correlation 3.")
        corr_matrix_no_motif = None
        n_eff_no_motif = 0
    else:

        # Compute correlation 3 (same logic as original function)
        pair_centers_no_motif = center_no_motif_pairs[:, 0]
        pair_neighbors_no_motif = center_no_motif_pairs[:, 1]

        neighbor_no_motif_cell_types = sq_obj.labels[pair_neighbors_no_motif]
        
        # Use helper function for correlation computation (paired data)
        corr_matrix_no_motif, n_eff_no_motif = compute_cross_correlation_paired(
            expr_genes=expr_genes,
            pair_centers=pair_centers_no_motif,
            pair_neighbors=pair_neighbors_no_motif,
            center_mean=center_mean,
            cell_type_means=cell_type_means,
            neighbor_cell_types=neighbor_no_motif_cell_types,
            is_sparse=is_sparse
        )
        print(f"Unique center cells: {len(np.unique(pair_centers_no_motif))}")
        print(f"Unique neighbor cells: {len(np.unique(pair_neighbors_no_motif))}")
        print(f"Number of pairs: {len(pair_centers_no_motif)}")
        print(f"Effective sample size: {n_eff_no_motif}")

    end_time = time()
    print(f"Time for computing Correlation 3: {end_time - start_time:.2f} seconds")

    # ==================================================================================
    # Compute correlations for each cell type separately
    # ==================================================================================
    all_results = []

    for cell_type in non_center_types:
        print("\n" + "="*80)
        print(f"Processing cell type: {cell_type}")
        print("="*80)

        # Filter pairs and cells for this specific cell type
        pair_neighbors = center_neighbor_pairs[:, 1]
        neighbor_types = sq_obj.labels[pair_neighbors]
        type_mask = neighbor_types == cell_type

        if type_mask.sum() == 0:
            raise ValueError(f"Error: No neighbor pairs found for cell type {cell_type}. Shouldn't happen.")

        type_specific_pairs = center_neighbor_pairs[type_mask]
        type_specific_neighbor_cells = np.unique(type_specific_pairs[:, 1])
        type_specific_center_cells = np.unique(type_specific_pairs[:, 0])

        # Filter non-neighbor cells for this type
        type_non_neighbor_mask = sq_obj.labels[non_neighbor_cells] == cell_type
        type_non_neighbor_cells = non_neighbor_cells[type_non_neighbor_mask]

        if len(type_non_neighbor_cells) < 10:
            print(f"Not enough non-neighbor cells ({len(type_non_neighbor_cells)}) for {cell_type}. Skipping.")
            continue
        
        print(f"{len(type_specific_center_cells)} center cells paired with {cell_type}")
        print(f"{len(type_specific_neighbor_cells)} neighboring cells of {cell_type}")
        print(f"{len(type_non_neighbor_cells)} distant cells of {cell_type}")
        print(f"{len(type_specific_pairs)} center-neighboring {cell_type} pairs")

        # ==================================================================================
        # Correlation 1: Center with motif vs Neighboring cells of THIS TYPE (PAIRED)
        # ==================================================================================
        print(f"\nComputing Correlation-1...")
        start_time = time()

        pair_centers = type_specific_pairs[:, 0]
        pair_neighbors_idx = type_specific_pairs[:, 1]

        # All neighbors are of the same type, create uniform type array
        neighbor_types_uniform = np.full(len(pair_neighbors_idx), cell_type)

        corr_matrix_neighbor, n_eff_neighbor = compute_cross_correlation_paired(
            expr_genes=expr_genes,
            pair_centers=pair_centers,
            pair_neighbors=pair_neighbors_idx,
            center_mean=center_mean,
            cell_type_means=cell_type_means,
            neighbor_cell_types=neighbor_types_uniform,
            is_sparse=is_sparse
        )

        print(f"  Effective sample size: {n_eff_neighbor}")
        print(f"  Time: {time() - start_time:.2f} seconds")

        # ==================================================================================
        # Correlation 2: Center with motif vs Distant cells of THIS TYPE (ALL PAIRS)
        # ==================================================================================
        print(f"\nComputing Correlation-2 ...")
        start_time = time()

        n_non_neighbor = len(type_non_neighbor_cells)

        # All non-neighbors are of the same type, create uniform type array
        non_neighbor_types_uniform = np.full(n_non_neighbor, cell_type)

        corr_matrix_non_neighbor, n_eff_non_neighbor = compute_cross_correlation_all_to_all(
            expr_genes=expr_genes,
            center_cells=type_specific_center_cells,
            non_neighbor_cells=type_non_neighbor_cells,
            center_mean=center_mean,
            cell_type_means=cell_type_means,
            non_neighbor_cell_types=non_neighbor_types_uniform,
            is_sparse=is_sparse
        )

        print(f"  Effective sample size: {n_eff_non_neighbor}")
        print(f"  Time: {time() - start_time:.2f} seconds")

        # ==================================================================================
        # Statistical testing
        # ==================================================================================
        print(f"\nPerforming statistical tests for {cell_type}...")
        start_time = time()

        # Test 1: Corr1 vs Corr2
        _, p_value_test1 = fisher_z_test(
            corr_matrix_neighbor, n_eff_neighbor,
            corr_matrix_non_neighbor, n_eff_non_neighbor
        )
        delta_corr_test1 = corr_matrix_neighbor - corr_matrix_non_neighbor

        # Test 2: Corr1 vs Corr3
        if corr_matrix_no_motif is not None:
            _, p_value_test2 = fisher_z_test(
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
            'gene_center': np.array(filtered_genes)[gene_center_idx.flatten()],
            'gene_motif': np.array(filtered_genes)[gene_motif_idx.flatten()],
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

        print(f"  Time: {time() - start_time:.2f} seconds")

    # ==================================================================================
    # Combine results from all cell types and apply FDR correction
    # ==================================================================================
    print("\n" + "="*80)
    print("Combining results and applying FDR correction")
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
            combined_results = combined_results[same_direction]
            p_values_test1 = combined_results['p_value_test1'].values
            p_values_test2 = combined_results['p_value_test2'].values

            all_p_values = np.concatenate([p_values_test1, p_values_test2])
            n_consistent = same_direction.sum()

            print(f"Applying FDR correction to {len(all_p_values)} total tests (2 × {n_consistent} gene pairs)")

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

def compute_gene_gene_correlation_by_type_binary(sq_obj,
                                                 ct: str,
                                                 motif: Union[str, List[str]],
                                                 genes: Optional[Union[str, List[str]]] = None,
                                                 max_dist: Optional[float] = None,
                                                 k: Optional[int] = None,
                                                 min_size: int = 0,
                                                 min_nonzero: int = 10,
                                                 alpha: Optional[float] = None
                                                 ) -> pd.DataFrame:
    """
    Compute gene-gene correlation using deviation from global binary mean, separately for each cell type in the motif.

    For each non-center cell type in the motif, compute:
    - Correlation 1: Center cells with motif vs neighboring motif cells of THIS TYPE
    - Correlation 2: Center cells with motif vs distant motif cells of THIS TYPE
    - Correlation 3: Center cells without motif vs neighbors (same for all types)

    Only analyzes motifs with >= 2 cell types besides the center type.

    Parameters
    ----------
    ct : str
        Cell type as the center cells.
    motif : Union[str, List[str]]
        Motif (names of cell types) to be analyzed.
    genes : Optional[Union[str, List[str]]], default=None
        List of genes to analyze. If None, all genes in the index will be used.
    max_dist : Optional[float], default=None
        Maximum distance for considering a cell as a neighbor. Use either max_dist or k.
    k : Optional[int], default=None
        Number of nearest neighbors. Use either max_dist or k.
    min_size : int, default=0
        Minimum neighborhood size for each center cell.
    min_nonzero : int, default=10
        Minimum number of non-zero expression values required for a gene to be included.
    alpha:
        Significance threshold.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with correlation results for each cell type and gene pair.
        Columns include:
            - cell_type: the non-center cell type in motif
            - gene_center, gene_motif: gene pairs
            - corr_neighbor: correlation with neighboring cells of this type
            - corr_non_neighbor: correlation with distant cells of this type
            - corr_center_no_motif: correlation with neighbors when no motif present
            - p_value_test1, p_value_test2: p-values for statistical tests
            - q_value_test1, q_value_test2: FDR-corrected q-values
            - delta_corr_test1, delta_corr_test2: correlation differences
            - reject_test1_fdr, reject_test2_fdr: FDR significance flags
            - combined_score, abs_combined_score: combined significance scores
            - if_significant: whether both tests pass FDR threshold
    """
    motif = motif if isinstance(motif, list) else [motif]

    # Get non-center cell types in motif
    non_center_types = [m for m in motif if m != ct]

    if alpha is None:
        alpha = 0.1

    if len(non_center_types) == 1:
        print(f"Only one non-center cell type in motif: {non_center_types}. Using compute_gene_gene_correlation_binary method.")
        result, _ = compute_gene_gene_correlation_binary(
            sq_obj=sq_obj,
            ct=ct,
            motif=motif,
            genes=genes,
            max_dist=max_dist,
            k=k,
            min_size=min_size,
            min_nonzero=min_nonzero,
            alpha=alpha
        )
        result['cell_type'] = non_center_types[0]
        return result
    elif len(non_center_types) == 0:
        raise ValueError("Error: Only center cell type in motif. Please ensure motif includes at least one non-center cell type.")

    print(f"Analyzing {len(non_center_types)} non-center cell types in motif: {non_center_types}")
    print("="*80)

    # Get neighbor and non-neighbor cell IDs using original motif
    neighbor_result = spatial_utils.get_motif_neighbor_cells(
        sq_obj=sq_obj, ct=ct, motif=motif, max_dist=max_dist, k=k, min_size=min_size
    )

    center_neighbor_pairs = neighbor_result['center_neighbor_pairs']
    ct_in_motif = neighbor_result['ct_in_motif']

    # Extract unique center and neighbor cells from pairs
    center_cells = np.unique(center_neighbor_pairs[:, 0])
    neighbor_cells = np.unique(center_neighbor_pairs[:, 1])

    # Get all motif cells
    motif_mask = np.isin(sq_obj.labels, motif)
    all_motif_cells = np.where(motif_mask)[0]
    non_neighbor_cells = np.setdiff1d(all_motif_cells, neighbor_cells)

    # Remove center type cells from non-neighbor cells
    if ct_in_motif:
        center_cell_mask_non = sq_obj.labels[non_neighbor_cells] == ct
        non_neighbor_cells = non_neighbor_cells[~center_cell_mask_non]
        print(f'Removed {center_cell_mask_non.sum()} center type cells from non-neighbor group.')

    # Get gene list
    if genes is None:
        valid_genes = sq_obj.genes
    elif isinstance(genes, str):
        genes = [genes]
        valid_genes = [g for g in genes if g in sq_obj.index.scfindGenes]
    else:
        valid_genes = [g for g in genes if g in sq_obj.index.scfindGenes]

    if len(valid_genes) == 0:
        raise ValueError("No valid genes found in the scfind index.")

    print(f"Building binary expression matrix from scfind index for {len(valid_genes)} genes...")
    start_time = time()

    # Use efficient C++ method to get sparse matrix data directly
    sparse_data = sq_obj.index.index.getBinarySparseMatrixData(valid_genes, sq_obj.dataset, min_nonzero)

    rows = sparse_data['rows']
    cols = sparse_data['cols']
    filtered_genes = sparse_data['gene_names']
    n_cells = sparse_data['n_cells']

    if len(filtered_genes) == 0:
        raise ValueError(f"No genes passed the min_nonzero={min_nonzero} filter.")

    # Create binary sparse matrix (cells × genes)
    binary_expr = csr_matrix(
        (np.ones(len(rows),), (rows, cols)),
        shape=(n_cells, len(filtered_genes)),
    )
    print(f"Time for recovering binary matrix: {time() - start_time:.2f} seconds")

    # Compute global binary means for ALL cell types
    print("\n" + "="*80)
    print("Computing global binary expression means for all cell types")
    print("="*80)
    start_time = time()

    all_cell_types = np.unique(sq_obj.labels)
    cell_type_means = {}

    for cell_type in all_cell_types:
        ct_mask = sq_obj.labels == cell_type
        ct_cells = np.where(ct_mask)[0]
        if len(ct_cells) > 0:
            ct_expr = binary_expr[ct_cells, :]
            cell_type_means[cell_type] = np.array(ct_expr.mean(axis=0)).flatten()

    center_mean = cell_type_means[ct]
    n_genes = len(filtered_genes)
    print(f"\nTime for computing global means: {time() - start_time:.2f} seconds")

    # ==================================================================================
    # Correlation 3: Center without motif vs Neighbors (SAME FOR ALL TYPES)
    # ==================================================================================
    print("\n" + "="*80)
    print("Computing Correlation-3: Center without motif vs Neighbors")
    print("="*80)

    start_time = time()
    no_motif_result = spatial_utils.get_all_neighbor_cells(
        sq_obj=sq_obj,
        ct=ct,
        max_dist=max_dist,
        k=k,
        min_size=min_size,
        exclude_centers=center_cells,
        exclude_neighbors=neighbor_cells,
    )

    center_no_motif_pairs = no_motif_result['center_neighbor_pairs']

    if len(center_no_motif_pairs) < 10:
        print(f"Not enough pairs ({len(center_no_motif_pairs)}) for centers without motif. Skipping Correlation 3.")
        corr_matrix_no_motif = None
        n_eff_no_motif = 0
    else:
        pair_centers_no_motif = center_no_motif_pairs[:, 0]
        pair_neighbors_no_motif = center_no_motif_pairs[:, 1]

        neighbor_no_motif_cell_types = sq_obj.labels[pair_neighbors_no_motif]

        corr_matrix_no_motif, n_eff_no_motif = compute_cross_correlation_paired(
            expr_genes=binary_expr,
            pair_centers=pair_centers_no_motif,
            pair_neighbors=pair_neighbors_no_motif,
            center_mean=center_mean,
            cell_type_means=cell_type_means,
            neighbor_cell_types=neighbor_no_motif_cell_types,
            is_sparse=True
        )
        print(f"Unique center cells: {len(np.unique(pair_centers_no_motif))}")
        print(f"Unique neighbor cells: {len(np.unique(pair_neighbors_no_motif))}")
        print(f"Number of pairs: {len(pair_centers_no_motif)}")
        print(f"Effective sample size: {n_eff_no_motif}")

    end_time = time()
    print(f"Time for computing Correlation 3: {end_time - start_time:.2f} seconds")

    # ==================================================================================
    # Compute correlations for each cell type separately
    # ==================================================================================
    all_results = []

    for cell_type in non_center_types:
        print("\n" + "="*80)
        print(f"Processing cell type: {cell_type}")
        print("="*80)

        # Filter pairs and cells for this specific cell type
        pair_neighbors = center_neighbor_pairs[:, 1]
        neighbor_types = sq_obj.labels[pair_neighbors]
        type_mask = neighbor_types == cell_type

        if type_mask.sum() == 0:
            raise ValueError(f"Error: No neighbor pairs found for cell type {cell_type}. Shouldn't happen.")

        type_specific_pairs = center_neighbor_pairs[type_mask]
        type_specific_neighbor_cells = np.unique(type_specific_pairs[:, 1])
        type_specific_center_cells = np.unique(type_specific_pairs[:, 0])

        # Filter non-neighbor cells for this type
        type_non_neighbor_mask = sq_obj.labels[non_neighbor_cells] == cell_type
        type_non_neighbor_cells = non_neighbor_cells[type_non_neighbor_mask]

        if len(type_non_neighbor_cells) < 10:
            print(f"Not enough non-neighbor cells ({len(type_non_neighbor_cells)}) for {cell_type}. Skipping.")
            continue

        print(f"{len(type_specific_center_cells)} center cells paired with {cell_type}")
        print(f"{len(type_specific_neighbor_cells)} neighboring cells of {cell_type}")
        print(f"{len(type_non_neighbor_cells)} distant cells of {cell_type}")
        print(f"{len(type_specific_pairs)} center-neighboring {cell_type} pairs")

        # ==================================================================================
        # Correlation 1: Center with motif vs Neighboring cells of THIS TYPE (PAIRED)
        # ==================================================================================
        print(f"\nComputing Correlation-1...")
        start_time = time()

        pair_centers = type_specific_pairs[:, 0]
        pair_neighbors_idx = type_specific_pairs[:, 1]

        # All neighbors are of the same type, create uniform type array
        neighbor_types_uniform = np.full(len(pair_neighbors_idx), cell_type)

        corr_matrix_neighbor, n_eff_neighbor = compute_cross_correlation_paired(
            expr_genes=binary_expr,
            pair_centers=pair_centers,
            pair_neighbors=pair_neighbors_idx,
            center_mean=center_mean,
            cell_type_means=cell_type_means,
            neighbor_cell_types=neighbor_types_uniform,
            is_sparse=True
        )

        print(f"  Effective sample size: {n_eff_neighbor}")
        print(f"  Time: {time() - start_time:.2f} seconds")

        # ==================================================================================
        # Correlation 2: Center with motif vs Distant cells of THIS TYPE (ALL PAIRS)
        # ==================================================================================
        print(f"\nComputing Correlation-2...")
        start_time = time()

        n_non_neighbor = len(type_non_neighbor_cells)

        # All non-neighbors are of the same type
        non_neighbor_types_uniform = np.full(n_non_neighbor, cell_type)

        corr_matrix_non_neighbor, n_eff_non_neighbor = compute_cross_correlation_all_to_all(
            expr_genes=binary_expr,
            center_cells=type_specific_center_cells,
            non_neighbor_cells=type_non_neighbor_cells,
            center_mean=center_mean,
            cell_type_means=cell_type_means,
            non_neighbor_cell_types=non_neighbor_types_uniform,
            is_sparse=True
        )

        print(f"  Effective sample size: {n_eff_non_neighbor}")
        print(f"  Time: {time() - start_time:.2f} seconds")

        # ==================================================================================
        # Statistical testing
        # ==================================================================================
        print(f"\nPerforming statistical tests for {cell_type}...")
        start_time = time()

        # Test 1: Corr1 vs Corr2
        _, p_value_test1 = fisher_z_test(
            corr_matrix_neighbor, n_eff_neighbor,
            corr_matrix_non_neighbor, n_eff_non_neighbor
        )
        delta_corr_test1 = corr_matrix_neighbor - corr_matrix_non_neighbor

        # Test 2: Corr1 vs Corr3
        if corr_matrix_no_motif is not None:
            _, p_value_test2 = fisher_z_test(
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
            'gene_center': np.array(filtered_genes)[gene_center_idx.flatten()],
            'gene_motif': np.array(filtered_genes)[gene_motif_idx.flatten()],
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

        print(f"  Time: {time() - start_time:.2f} seconds")

    # ==================================================================================
    # Combine results from all cell types and apply FDR correction
    # ==================================================================================
    print("\n" + "="*80)
    print("Combining results and applying FDR correction")
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
            combined_results = combined_results[same_direction]
            p_values_test1 = combined_results['p_value_test1'].values
            p_values_test2 = combined_results['p_value_test2'].values

            all_p_values = np.concatenate([p_values_test1, p_values_test2])
            n_consistent = same_direction.sum()

            print(f"Applying FDR correction to {len(all_p_values)} total tests (2 × {n_consistent} gene pairs)")
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

    print(f"Significant pairs in A: {len(sig_pairs_A)}")
    print(f"Significant pairs in B: {len(sig_pairs_B)}")
    print(f"Pairs that are significant in at least one group: {len(at_least_one_sig)}")

    # Find overlapping pairs (present in both groups)
    pairs_A = set(result_A['pair_id'])
    pairs_B = set(result_B['pair_id'])
    overlapping_pairs = pairs_A & pairs_B

    # Filter for overlapping pairs that are significant in at least one group

    pairs_to_test = overlapping_pairs & at_least_one_sig if background == 'Significant' else overlapping_pairs
    # pairs_to_test = overlapping_pairs  # TODO: 先试试如果把所有overlapping的pairs作为null distribution有什么影响. 返回的结果肯定更多


    if len(pairs_to_test) == 0:
        raise ValueError("No overlapping gene-pairs found that are significant in at least one group")

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
    print(f"  Higher in group A: {n_higher_A}")
    print(f"  Lower in group A: {n_lower_A}")
    print(f"\nScore difference range: [{result['score_diff'].min():.3f}, {result['score_diff'].max():.3f}]")
    print(f"Mean score difference: {result['score_diff'].mean():.3f}")
    print(f"Std score difference: {result['score_diff'].std():.3f}")

    return result


def compute_gene_gene_correlation_adata_multi_fov(
        sq_objs,
        ct: str,
        motif: Union[str, List[str]],
        dataset: Union[str, List[str]] = None,
        genes: Optional[Union[str, List[str]]] = None,
        max_dist: Optional[float] = None,
        k: Optional[int] = None,
        min_size: int = 0,
        min_nonzero: int = 10,
        alpha: Optional[float] = None
        ) -> pd.DataFrame:
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
    alpha: 
        Significance threshold.

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
    
    if alpha is None:
        alpha = 0.05

    # Convert motif to list
    motif = motif if isinstance(motif, list) else [motif]

    # Validate and prepare dataset list
    if dataset is None:
        dataset = [s.dataset.split('_')[0] for s in sq_objs.spatial_queries]
        print(f"No dataset specified. Using all datasets.")
    if isinstance(dataset, str):
        dataset = [dataset]

    valid_ds_names = [s.dataset.split('_')[0] for s in sq_objs.spatial_queries]
    for ds in dataset:
        if ds not in valid_ds_names:
            raise ValueError(f"Invalid input dataset name: {ds}.\n "
                            f"Valid dataset names are: {set(valid_ds_names)}")

    # Filter spatial_queries to include only selected datasets
    selected_queries = [s for s in sq_objs.spatial_queries if s.dataset.split('_')[0] in dataset]

    # Check if ct and motif exist in at least one FOV
    ct_exists = any(ct in s.labels.unique() for s in selected_queries)
    if not ct_exists:
        raise ValueError(f"Center type '{ct}' not found in any selected datasets!")

    motif_exists = all(any(m in s.labels.unique() for s in selected_queries) for m in motif)
    if not motif_exists:
        missing = [m for m in motif if not any(m in s.labels.unique() for s in selected_queries)]
        raise ValueError(f"Motif types {missing} not found in any selected datasets!")

    # Get union of genes across all FOVs
    genes_sets = [set(sq.genes) for sq in selected_queries]
    all_genes_union = list(set.union(*genes_sets))
    all_genes_intersection = list(set.intersection(*genes_sets))

    if genes is None:
        print(f"No genes specified. Using union of genes across all selected FOVs ...")
        valid_genes = all_genes_union
    elif isinstance(genes, str):
        genes = [genes]
        valid_genes = [g for g in genes if g in all_genes_union]
    else:
        valid_genes = [g for g in genes if g in all_genes_union]

    if len(valid_genes) == 0:
        raise ValueError("No valid genes found in any FOV.")

    genes = valid_genes
    n_genes = len(genes)

    # Report gene coverage statistics
    n_union = len(all_genes_union)
    n_intersection = len(all_genes_intersection)
    n_excluded = n_union - n_intersection
    print(f"Gene coverage: {n_intersection} genes in all FOVs, {n_union} genes total (union)")
    if n_excluded > 0:
        print(f"  -> {n_excluded} genes present in subset of FOVs (will use available data)")
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

        # Find which genes from our gene list exist in this FOV
        fov_genes_set = set(sq.genes)
        genes_in_fov = [g for g in genes if g in fov_genes_set]

        if len(genes_in_fov) == 0:
            print(f"  Skipping: no genes from analysis list found in this FOV")
            continue

        # Create index mapping from FOV genes to full gene list
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        fov_gene_indices = np.array([gene_to_idx[g] for g in genes_in_fov])

        n_genes_fov = len(genes_in_fov)
        print(f"  FOV has {n_genes_fov}/{n_genes} genes from analysis list")

        # Get expression data for this FOV (only genes present in this FOV)
        expr_genes = sq.adata[:, genes_in_fov].X
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
                cov_sum, center_ss, neighbor_ss, n_pairs, n_eff = compute_covariance_statistics_paired(
                    expr_genes=expr_genes,
                    pair_centers=pair_centers,
                    pair_neighbors=pair_neighbors,
                    center_mean=center_mean,
                    cell_type_means=fov_cell_type_means,
                    neighbor_cell_types=neighbor_types,
                    is_sparse=is_sparse
                )

                # Map FOV-specific statistics to full gene indices
                # cov_sum is (n_genes_fov, n_genes_fov), need to map to (n_genes, n_genes)
                ix = np.ix_(fov_gene_indices, fov_gene_indices)
                total_cov_sum_neighbor[ix] += cov_sum
                total_center_ss_neighbor[fov_gene_indices] += center_ss
                total_neighbor_ss_neighbor[fov_gene_indices] += neighbor_ss
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
                    cov_sum_non, center_ss_non, non_neighbor_ss, n_pairs_non, n_eff_non = compute_covariance_statistics_all_to_all(
                        expr_genes=expr_genes,
                        center_cells=unique_centers,
                        neighbor_cells=non_neighbor_cells,
                        center_mean=center_mean,
                        cell_type_means=fov_cell_type_means,
                        neighbor_cell_types=non_neighbor_types,
                        is_sparse=is_sparse
                    )

                    # Map FOV-specific statistics to full gene indices
                    ix = np.ix_(fov_gene_indices, fov_gene_indices)
                    total_cov_sum_non[ix] += cov_sum_non
                    total_center_ss_non[fov_gene_indices] += center_ss_non
                    total_non_neighbor_ss[fov_gene_indices] += non_neighbor_ss
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
                    cov_sum_no_motif, center_ss_no_motif, neighbor_ss_no_motif, n_pairs_no_motif, n_eff_no_motif = compute_covariance_statistics_paired(
                        expr_genes=expr_genes,
                        pair_centers=pair_centers_no_motif,
                        pair_neighbors=pair_neighbors_no_motif,
                        center_mean=center_mean,
                        cell_type_means=fov_cell_type_means,
                        neighbor_cell_types=neighbor_no_motif_types,
                        is_sparse=is_sparse
                    )

                    # Map FOV-specific statistics to full gene indices
                    ix = np.ix_(fov_gene_indices, fov_gene_indices)
                    total_cov_sum_no_motif[ix] += cov_sum_no_motif
                    total_center_ss_no_motif[fov_gene_indices] += center_ss_no_motif
                    total_neighbor_ss_no_motif[fov_gene_indices] += neighbor_ss_no_motif
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
        _, p_value_test1 = fisher_z_test(
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
        _, p_value_test2 = fisher_z_test(
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
            sig_both = (adj_p_test1 < alpha) & (adj_p_test2 < alpha)
            sig_mask[sig_indices] = sig_both

            results_df.loc[sig_mask, 'if_significant'] = True

            print(f"Significant gene pairs (both tests, FDR < {alpha}): {sig_mask.sum()}")
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
        print(f"Significant gene pairs (test1, FDR < {alpha}): {rejected.sum()}")

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

def compute_gene_gene_correlation_binary_multi_fov(
        sq_objs,
        ct: str,
        motif: Union[str, List[str]],
        dataset: Union[str, List[str]] = None,
        genes: Optional[Union[str, List[str]]] = None,
        max_dist: Optional[float] = None,
        k: Optional[int] = None,
        min_size: int = 0,
        min_nonzero: int = 10,
        alpha: Optional[float] = None
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
    alpha: 
        Significance threshold.

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
    
    if alpha is None:
        alpha = 0.1

    # Convert motif to list
    motif = motif if isinstance(motif, list) else [motif]

    # Validate and prepare dataset list
    if dataset is None:
        dataset = [s.dataset.split('_')[0] for s in sq_objs.spatial_queries]
        print(f"No dataset specified. Using all datasets.")
    if isinstance(dataset, str):
        dataset = [dataset]

    valid_ds_names = [s.dataset.split('_')[0] for s in sq_objs.spatial_queries]
    for ds in dataset:
        if ds not in valid_ds_names:
            raise ValueError(f"Invalid input dataset name: {ds}.\n "
                            f"Valid dataset names are: {set(valid_ds_names)}")

    # Filter spatial_queries to include only selected datasets
    selected_queries = [s for s in sq_objs.spatial_queries if s.dataset.split('_')[0] in dataset]

    # Check if ct and motif exist in at least one FOV
    ct_exists = any(ct in s.labels.unique() for s in selected_queries)
    if not ct_exists:
        raise ValueError(f"Center type '{ct}' not found in any selected datasets!")

    motif_exists = all(any(m in s.labels.unique() for s in selected_queries) for m in motif)
    if not motif_exists:
        missing = [m for m in motif if not any(m in s.labels.unique() for s in selected_queries)]
        raise ValueError(f"Motif types {missing} not found in any selected datasets!")

    # Get union of genes across all FOVs
    genes_sets = [set(sq.genes) for sq in selected_queries]
    all_genes_union = list(set.union(*genes_sets))
    all_genes_intersection = list(set.intersection(*genes_sets))

    if genes is None:
        print(f"No genes specified. Using union of genes across all selected FOVs ...")
        valid_genes = all_genes_union
    elif isinstance(genes, str):
        genes = [genes]
        valid_genes = [g for g in genes if g in all_genes_union]
    else:
        valid_genes = [g for g in genes if g in all_genes_union]

    if len(valid_genes) == 0:
        raise ValueError("No valid genes found in any FOV.")

    genes = valid_genes
    n_genes = len(genes)

    # Report gene coverage statistics
    n_union = len(all_genes_union)
    n_intersection = len(all_genes_intersection)
    n_excluded = n_union - n_intersection
    print(f"Gene coverage: {n_intersection} genes in all FOVs, {n_union} genes total (union)")
    if n_excluded > 0:
        print(f"  -> {n_excluded} genes present in subset of FOVs (will use available data)")
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

        # Create index mapping from FOV genes to full gene list
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        fov_gene_indices = np.array([gene_to_idx[g] for g in filtered_genes])

        n_genes_fov = len(filtered_genes)
        print(f"  FOV has {n_genes_fov}/{n_genes} genes from analysis list")

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
                cov_sum, center_ss, neighbor_ss, n_pairs, n_eff = compute_covariance_statistics_paired(
                    expr_genes=binary_expr,
                    pair_centers=pair_centers,
                    pair_neighbors=pair_neighbors,
                    center_mean=center_mean,
                    cell_type_means=fov_cell_type_means,
                    neighbor_cell_types=neighbor_types,
                    is_sparse=is_sparse
                )

                # Map FOV-specific statistics to full gene indices
                # cov_sum is (n_genes_fov, n_genes_fov), need to map to (n_genes, n_genes)
                ix = np.ix_(fov_gene_indices, fov_gene_indices)
                total_cov_sum_neighbor[ix] += cov_sum
                total_center_ss_neighbor[fov_gene_indices] += center_ss
                total_neighbor_ss_neighbor[fov_gene_indices] += neighbor_ss
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
                    cov_sum_non, center_ss_non, non_neighbor_ss, n_pairs_non, n_eff_non = compute_covariance_statistics_all_to_all(
                        expr_genes=binary_expr,
                        center_cells=unique_centers,
                        neighbor_cells=non_neighbor_cells,
                        center_mean=center_mean,
                        cell_type_means=fov_cell_type_means,
                        neighbor_cell_types=non_neighbor_types,
                        is_sparse=is_sparse
                    )

                    # Map FOV-specific statistics to full gene indices
                    ix = np.ix_(fov_gene_indices, fov_gene_indices)
                    total_cov_sum_non[ix] += cov_sum_non
                    total_center_ss_non[fov_gene_indices] += center_ss_non
                    total_non_neighbor_ss[fov_gene_indices] += non_neighbor_ss
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
                    cov_sum_no_motif, center_ss_no_motif, neighbor_ss_no_motif, n_pairs_no_motif, n_eff_no_motif = compute_covariance_statistics_paired(
                        expr_genes=binary_expr,
                        pair_centers=pair_centers_no_motif,
                        pair_neighbors=pair_neighbors_no_motif,
                        center_mean=center_mean,
                        cell_type_means=fov_cell_type_means,
                        neighbor_cell_types=neighbor_no_motif_types,
                        is_sparse=is_sparse
                    )

                    # Map FOV-specific statistics to full gene indices
                    ix = np.ix_(fov_gene_indices, fov_gene_indices)
                    total_cov_sum_no_motif[ix] += cov_sum_no_motif
                    total_center_ss_no_motif[fov_gene_indices] += center_ss_no_motif
                    total_neighbor_ss_no_motif[fov_gene_indices] += neighbor_ss_no_motif
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
        _, p_value_test1 = fisher_z_test(
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
        _, p_value_test2 = fisher_z_test(
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
            sig_both = (adj_p_test1 < alpha) & (adj_p_test2 < alpha)
            sig_mask[sig_indices] = sig_both

            results_df.loc[sig_mask, 'if_significant'] = True

            print(f"Significant gene pairs (both tests, FDR < {alpha}): {sig_mask.sum()}")
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
        print(f"Significant gene pairs (test1, FDR < {alpha}): {rejected.sum()}")

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


def compute_gene_gene_correlation_by_type_adata_multi_fov(
        sq_objs,
        ct: str,
        motif: Union[str, List[str]],
        dataset: Union[str, List[str]] = None,
        genes: Optional[Union[str, List[str]]] = None,
        max_dist: Optional[float] = None,
        k: Optional[int] = None,
        min_size: int = 0,
        min_nonzero: int = 10,
        alpha: Optional[float] = None,
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
    alpha:
        Significance threshold.

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
    if alpha is None:
        alpha = 0.05

    if len(non_center_types) == 1:
        print(f"Only one non-center cell type in motif: {non_center_types}. Using compute_gene_gene_correlation method.")
        result, _ = compute_gene_gene_correlation_adata_multi_fov(
            sq_objs=sq_objs,
            ct=ct,
            motif=motif,
            dataset=dataset,
            genes=genes,
            max_dist=max_dist,
            k=k,
            min_size=min_size,
            min_nonzero=min_nonzero,
            alpha=alpha
        )
        result['cell_type'] = non_center_types[0]
        return result
    elif len(non_center_types) == 0:
        raise ValueError("Error: Only center cell type in motif. Please ensure motif includes at least one non-center cell type.")

    print(f"Analyzing {len(non_center_types)} non-center cell types in motif: {non_center_types}")
    print("="*80)

    # Select FOVs (handle dataset names with and without suffix)
    if dataset is None:
        dataset = [s.dataset.split('_')[0] for s in sq_objs.spatial_queries]
        print(f"No dataset specified. Using all datasets.")
    if isinstance(dataset, str):
        dataset = [dataset]

    valid_ds_names = [s.dataset.split('_')[0] for s in sq_objs.spatial_queries]
    for ds in dataset:
        if ds not in valid_ds_names:
            raise ValueError(f"Invalid input dataset name: {ds}.\n "
                            f"Valid dataset names are: {set(valid_ds_names)}")

    # Filter queries to include only selected datasets
    selected_queries = [s for s in sq_objs.spatial_queries if s.dataset.split('_')[0] in dataset]

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

    # Get union of genes across all FOVs
    genes_sets = [set(sq.genes) for sq in selected_queries]
    all_genes_union = list(set.union(*genes_sets))
    all_genes_intersection = list(set.intersection(*genes_sets))

    if genes is None:
        print(f"No genes specified. Using union of genes across all selected FOVs ...")
        valid_genes = all_genes_union
    elif isinstance(genes, str):
        genes = [genes]
        valid_genes = [g for g in genes if g in all_genes_union]
    else:
        valid_genes = [g for g in genes if g in all_genes_union]

    if len(valid_genes) == 0:
        raise ValueError("No valid genes found in any FOV.")

    genes = valid_genes
    n_genes = len(genes)

    # Report gene coverage statistics
    n_union = len(all_genes_union)
    n_intersection = len(all_genes_intersection)
    n_excluded = n_union - n_intersection
    print(f"Gene coverage: {n_intersection} genes in all FOVs, {n_union} genes total (union)")
    if n_excluded > 0:
        print(f"  -> {n_excluded} genes present in subset of FOVs (will use available data)")
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

        # Find which genes from our gene list exist in this FOV
        fov_genes_set = set(sq.genes)
        genes_in_fov = [g for g in genes if g in fov_genes_set]

        if len(genes_in_fov) == 0:
            print(f"  Skipping: no genes from analysis list found in this FOV")
            continue

        # Create index mapping from FOV genes to full gene list
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        fov_gene_indices = np.array([gene_to_idx[g] for g in genes_in_fov])

        n_genes_fov = len(genes_in_fov)
        print(f"  FOV has {n_genes_fov}/{n_genes} genes from analysis list")

        # Get expression data (only genes present in this FOV)
        expr_genes = sq.adata[:, genes_in_fov].X
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
                'fov_gene_indices': fov_gene_indices,  # Add gene mapping
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

                cov_sum, center_ss, neighbor_ss, n_pairs, n_eff = compute_covariance_statistics_paired(
                    expr_genes=expr_genes,
                    pair_centers=pair_centers_no_motif,
                    pair_neighbors=pair_neighbors_no_motif,
                    center_mean=center_mean,
                    cell_type_means=fov_cell_type_means,
                    neighbor_cell_types=neighbor_no_motif_types,
                    is_sparse=is_sparse
                )

                # Map FOV-specific statistics to full gene indices
                ix = np.ix_(fov_gene_indices, fov_gene_indices)
                total_cov_sum_no_motif[ix] += cov_sum
                total_center_ss_no_motif[fov_gene_indices] += center_ss
                total_neighbor_ss_no_motif[fov_gene_indices] += neighbor_ss
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
            fov_gene_indices = fov_data['fov_gene_indices']  # Get gene mapping

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

            cov_sum, center_ss, neighbor_ss, n_pairs, n_eff = compute_covariance_statistics_paired(
                expr_genes=expr_genes,
                pair_centers=pair_centers,
                pair_neighbors=pair_neighbors_idx,
                center_mean=center_mean,
                cell_type_means=fov_cell_type_means,
                neighbor_cell_types=neighbor_types_uniform,
                is_sparse=is_sparse
            )

            # Map FOV-specific statistics to full gene indices
            ix = np.ix_(fov_gene_indices, fov_gene_indices)
            total_cov_sum_neighbor[ix] += cov_sum
            total_center_ss_neighbor[fov_gene_indices] += center_ss
            total_neighbor_ss_neighbor[fov_gene_indices] += neighbor_ss
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

            cov_sum_non, center_ss_non, non_neighbor_ss, n_pairs_non, n_eff_non = compute_covariance_statistics_all_to_all(
                expr_genes=expr_genes,
                center_cells=type_specific_center_cells,
                neighbor_cells=type_non_neighbor_cells,
                center_mean=center_mean,
                cell_type_means=fov_cell_type_means,
                neighbor_cell_types=non_neighbor_types_uniform,
                is_sparse=is_sparse
            )

            # Map FOV-specific statistics to full gene indices
            ix = np.ix_(fov_gene_indices, fov_gene_indices)
            total_cov_sum_non[ix] += cov_sum_non
            total_center_ss_non[fov_gene_indices] += center_ss_non
            total_non_neighbor_ss[fov_gene_indices] += non_neighbor_ss
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
        _, p_value_test1 = fisher_z_test(
            corr_matrix_neighbor, n_eff_neighbor,
            corr_matrix_non_neighbor, n_eff_non_neighbor
        )
        delta_corr_test1 = corr_matrix_neighbor - corr_matrix_non_neighbor

        # Test 2: Corr1 vs Corr3
        if corr_matrix_no_motif is not None:
            _, p_value_test2 = fisher_z_test(
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

def compute_gene_gene_correlation_by_type_binary_multi_fov(
        sq_objs,
        ct: str,
        motif: Union[str, List[str]],
        dataset: Union[str, List[str]] = None,
        genes: Optional[Union[str, List[str]]] = None,
        max_dist: Optional[float] = None,
        k: Optional[int] = None,
        min_size: int = 0,
        min_nonzero: int = 10,
        alpha: Optional[float] = None,
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
        alpha:
            Significance threshold.

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
        if alpha is None:
            alpha = 0.1

        if len(non_center_types) == 1:
            print(f"Only one non-center cell type in motif: {non_center_types}. Using compute_gene_gene_correlation_binary method.")
            result = compute_gene_gene_correlation_binary_multi_fov(
                ct=ct,
                motif=motif,
                dataset=dataset,
                genes=genes,
                max_dist=max_dist,
                k=k,
                min_size=min_size,
                min_nonzero=min_nonzero,
                alpha=alpha
            )
            result['cell_type'] = non_center_types[0]
            return result
        elif len(non_center_types) == 0:
            raise ValueError("Error: Only center cell type in motif. Please ensure motif includes at least one non-center cell type.")

        print(f"Analyzing {len(non_center_types)} non-center cell types in motif: {non_center_types}")
        print("="*80)

        # Select FOVs (handle dataset names with and without suffix)
        if dataset is None:
            dataset = [s.dataset.split('_')[0] for s in sq_objs.spatial_queries]
            print(f"No dataset specified. Using all datasets.")
        if isinstance(dataset, str):
            dataset = [dataset]

        valid_ds_names = [s.dataset.split('_')[0] for s in sq_objs.spatial_queries]
        for ds in dataset:
            if ds not in valid_ds_names:
                raise ValueError(f"Invalid input dataset name: {ds}.\n "
                               f"Valid dataset names are: {set(valid_ds_names)}")

        # Filter queries to include only selected datasets
        selected_queries = [s for s in sq_objs.spatial_queries if s.dataset.split('_')[0] in dataset]

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

        # Get union of genes across all FOVs
        genes_sets = [set(sq.genes) for sq in selected_queries]
        all_genes_union = list(set.union(*genes_sets))
        all_genes_intersection = list(set.intersection(*genes_sets))

        if genes is None:
            print(f"No genes specified. Using union of genes across all selected FOVs ...")
            valid_genes = all_genes_union
        elif isinstance(genes, str):
            genes = [genes]
            valid_genes = [g for g in genes if g in all_genes_union]
        else:
            valid_genes = [g for g in genes if g in all_genes_union]

        if len(valid_genes) == 0:
            raise ValueError("No valid genes found in any FOV.")

        genes = valid_genes
        n_genes = len(genes)

        # Report gene coverage statistics
        n_union = len(all_genes_union)
        n_intersection = len(all_genes_intersection)
        n_excluded = n_union - n_intersection
        print(f"Gene coverage: {n_intersection} genes in all FOVs, {n_union} genes total (union)")
        if n_excluded > 0:
            print(f"  -> {n_excluded} genes present in subset of FOVs (will use available data)")
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

            # Create index mapping from FOV genes to full gene list
            gene_to_idx = {g: i for i, g in enumerate(genes)}
            fov_gene_indices = np.array([gene_to_idx[g] for g in filtered_genes])

            n_genes_fov = len(filtered_genes)
            print(f"  FOV has {n_genes_fov}/{n_genes} genes from analysis list")

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
                    'fov_gene_indices': fov_gene_indices,  # Add gene mapping
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

                    cov_sum, center_ss, neighbor_ss, n_pairs, n_eff = compute_covariance_statistics_paired(
                        expr_genes=binary_expr,
                        pair_centers=pair_centers_no_motif,
                        pair_neighbors=pair_neighbors_no_motif,
                        center_mean=center_mean,
                        cell_type_means=fov_cell_type_means,
                        neighbor_cell_types=neighbor_no_motif_types,
                        is_sparse=is_sparse
                    )

                    # Map FOV-specific statistics to full gene indices
                    ix = np.ix_(fov_gene_indices, fov_gene_indices)
                    total_cov_sum_no_motif[ix] += cov_sum
                    total_center_ss_no_motif[fov_gene_indices] += center_ss
                    total_neighbor_ss_no_motif[fov_gene_indices] += neighbor_ss
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
                fov_gene_indices = fov_data['fov_gene_indices']  # Get gene mapping

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

                cov_sum, center_ss, neighbor_ss, n_pairs, n_eff = compute_covariance_statistics_paired(
                    expr_genes=expr_genes,
                    pair_centers=pair_centers,
                    pair_neighbors=pair_neighbors_idx,
                    center_mean=center_mean,
                    cell_type_means=fov_cell_type_means,
                    neighbor_cell_types=neighbor_types_uniform,
                    is_sparse=is_sparse
                )

                # Map FOV-specific statistics to full gene indices
                ix = np.ix_(fov_gene_indices, fov_gene_indices)
                total_cov_sum_neighbor[ix] += cov_sum
                total_center_ss_neighbor[fov_gene_indices] += center_ss
                total_neighbor_ss_neighbor[fov_gene_indices] += neighbor_ss
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

                cov_sum_non, center_ss_non, non_neighbor_ss, n_pairs_non, n_eff_non = compute_covariance_statistics_all_to_all(
                    expr_genes=expr_genes,
                    center_cells=type_specific_center_cells,
                    neighbor_cells=type_non_neighbor_cells,
                    center_mean=center_mean,
                    cell_type_means=fov_cell_type_means,
                    neighbor_cell_types=non_neighbor_types_uniform,
                    is_sparse=is_sparse
                )

                # Map FOV-specific statistics to full gene indices
                ix = np.ix_(fov_gene_indices, fov_gene_indices)
                total_cov_sum_non[ix] += cov_sum_non
                total_center_ss_non[fov_gene_indices] += center_ss_non
                total_non_neighbor_ss[fov_gene_indices] += non_neighbor_ss
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
            _, p_value_test1 = fisher_z_test(
                corr_matrix_neighbor, n_eff_neighbor,
                corr_matrix_non_neighbor, n_eff_non_neighbor
            )
            delta_corr_test1 = corr_matrix_neighbor - corr_matrix_non_neighbor

            # Test 2: Corr1 vs Corr3
            if corr_matrix_no_motif is not None:
                _, p_value_test2 = fisher_z_test(
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