"""
Functions for differential pattern analysis across multiple datasets.
"""

from typing import List, Union, Tuple, Dict
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multitest as mt
from sklearn.preprocessing import LabelEncoder

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


@njit
def _compute_motif_support_numba(neighbor_matrix, motif_indices_array, motif_lengths,
                                  has_valid_size, num_center_cells):
    """
    Numba-accelerated function to compute support for multiple motifs.

    Parameters
    ----------
    neighbor_matrix : np.ndarray, shape (num_center_cells, num_types)
        Binary matrix indicating presence of each cell type in each center cell's neighborhood
    motif_indices_array : np.ndarray, shape (num_motifs, max_motif_len)
        Integer indices of cell types for each motif (padded with -1)
    motif_lengths : np.ndarray, shape (num_motifs,)
        Actual length of each motif
    has_valid_size : np.ndarray, shape (num_center_cells,)
        Boolean mask for cells with valid neighborhood size
    num_center_cells : int
        Number of center cells

    Returns
    -------
    supports : np.ndarray, shape (num_motifs,)
        Support value for each motif
    """
    num_motifs = motif_indices_array.shape[0]
    supports = np.zeros(num_motifs, dtype=np.float64)

    for motif_idx in range(num_motifs):
        motif_len = motif_lengths[motif_idx]
        motif_count = 0

        # For each center cell
        for cell_idx in range(num_center_cells):
            # Check if this cell has valid neighborhood size
            if not has_valid_size[cell_idx]:
                continue

            # Check if all motif cell types are present
            has_all_types = True
            for type_idx in range(motif_len):
                cell_type = motif_indices_array[motif_idx, type_idx]
                if neighbor_matrix[cell_idx, cell_type] == 0:
                    has_all_types = False
                    break

            if has_all_types:
                motif_count += 1

        supports[motif_idx] = motif_count / num_center_cells

    return supports


def differential_analysis_motif_knn(
    spatial_queries: List,
    datasets_list: List[str],
    ct: str,
    motifs: Union[str, List[str], List[List[str]]],
    k: int = 30,
    max_dist: float = 20,
) -> Dict[str, pd.DataFrame]:
    """
    Test whether user-specified motifs are differentially enriched across two datasets
    using k-nearest neighbors approach.

    Parameters
    ----------
    spatial_queries : List
        List of spatial_query objects from spatial_query_multi
    datasets_list : List[str]
        Two dataset names for differential analysis (length must be 2)
    ct : str
        Cell type of interest as center point
    motifs : Union[str, List[str], List[List[str]]]
        User-specified motif(s) to test. Can be:
        - Single cell type: 'CellTypeA'
        - Single motif: ['CellTypeA', 'CellTypeB']
        - Multiple motifs: [['CellTypeA'], ['CellTypeB', 'CellTypeC']]
    k : int, default=30
        Number of nearest neighbors
    max_dist : float, default=20
        Maximum distance for considering a cell as a neighbor

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys as dataset names, values as DataFrames with significant enriched motifs.
        Each DataFrame contains:
        - itemsets: the tested motif pattern (as tuple)
        - adj_pvals: FDR-corrected p-value
        Only significant motifs (adj_p_value < 0.05) for each dataset are included.
    """
    if len(datasets_list) != 2:
        raise ValueError("Require 2 datasets for differential analysis.")

    # Normalize motifs to list of lists
    if isinstance(motifs, str):
        # Single cell type as string: 'CellTypeA' -> [['CellTypeA']]
        motifs = [[motifs]]
    elif isinstance(motifs, list) and len(motifs) > 0 and isinstance(motifs[0], str):
        # Single motif as list of strings: ['CellTypeA', 'CellTypeB'] -> [['CellTypeA', 'CellTypeB']]
        motifs = [motifs]
    # else: already in correct format [['CellTypeA'], ['CellTypeB', 'CellTypeC']]


    # Initialize support storage for all motifs
    # Structure: {motif_tuple: {dataset_name: [support_fov1, support_fov2, ...]}}
    motif_supports = {}
    for motif in motifs:
        motif_sorted = tuple(sorted(motif))
        motif_supports[motif_sorted] = {datasets_list[0]: [], datasets_list[1]: []}

    # Process each FOV and compute support for all motifs
    for dataset_name in datasets_list:
        # Get all FOVs for this dataset
        dataset_queries = [sq for sq in spatial_queries
                         if sq.dataset.split('_')[0] == dataset_name]

        for sq in dataset_queries:
            cell_pos = sq.spatial_pos
            labels = np.array(sq.labels)
        
            # Get center cell positions and indices
            ct_pos = cell_pos[labels == ct]
            cinds = np.where(labels == ct)[0]
            num_center_cells = len(cinds)
            
            # Check if center cell type exists
            if num_center_cells == 0:
                for motif_sorted in motif_supports:
                    motif_supports[motif_sorted][dataset_name].append(0.0)
                continue

            # Query KNN for center cells
            dists, idxs = sq.kd_tree.query(ct_pos, k=k + 1, workers=-1)

            # Vectorized computation: create neighbor count matrix
            label_encoder = LabelEncoder()
            int_labels = label_encoder.fit_transform(labels)

            # Filter neighbors by distance
            valid_neighbors = dists[:, 1:] <= max_dist  # Exclude self (index 0)
            filtered_idxs = np.where(valid_neighbors, idxs[:, 1:], -1)
            flat_neighbors = filtered_idxs.flatten()
            valid_neighbors_flat = valid_neighbors.flatten()

            # Get neighbor labels
            neighbor_labels = np.where(valid_neighbors_flat, int_labels[flat_neighbors], -1)
            valid_mask = neighbor_labels != -1

            # Create neighbor count matrix: [num_center_cells, num_cell_types]
            num_types = len(label_encoder.classes_)
            neighbor_matrix = np.zeros((num_center_cells * k, num_types), dtype=int)
            neighbor_matrix[np.arange(len(neighbor_labels))[valid_mask], neighbor_labels[valid_mask]] = 1
            neighbor_counts = neighbor_matrix.reshape(num_center_cells, k, num_types).sum(axis=1)

            # Now compute support for all motifs in this FOV - OPTIMIZED with Numba
            labels_unique = label_encoder.classes_
            labels_set = set(labels_unique)

            # Separate valid and invalid motifs
            valid_motifs_sorted = []
            valid_motifs_data = []

            for motif in motifs:
                motif_sorted = tuple(sorted(motif))

                # Quick check: all cell types exist in this FOV?
                if all(m in labels_set for m in motif):
                    # Convert to integer indices
                    int_motifs = label_encoder.transform(np.array(motif))
                    valid_motifs_sorted.append(motif_sorted)
                    valid_motifs_data.append(int_motifs)
                else:
                    # Motif cell types don't exist in this FOV
                    motif_supports[motif_sorted][dataset_name].append(0.0)

            # If there are valid motifs, use numba-accelerated batch processing
            if len(valid_motifs_sorted) > 0 and NUMBA_AVAILABLE:
                # Prepare data for numba function (KNN version: all neighborhoods have valid size)
                max_motif_len = max(len(m) for m in valid_motifs_data)
                motif_indices_array = np.full((len(valid_motifs_data), max_motif_len), -1, dtype=np.int32)
                motif_lengths = np.array([len(m) for m in valid_motifs_data], dtype=np.int32)

                for i, int_motifs in enumerate(valid_motifs_data):
                    motif_indices_array[i, :len(int_motifs)] = int_motifs

                # For KNN, all neighborhoods are valid (no min_size filter in this version)
                has_valid_size = np.ones(num_center_cells, dtype=np.bool_)

                # Call numba-accelerated function
                supports = _compute_motif_support_numba(
                    neighbor_counts.astype(np.int32),  # Use neighbor_counts for KNN
                    motif_indices_array,
                    motif_lengths,
                    has_valid_size,
                    num_center_cells
                )

                # Store results
                for motif_sorted, support in zip(valid_motifs_sorted, supports):
                    motif_supports[motif_sorted][dataset_name].append(support)

            elif len(valid_motifs_sorted) > 0:
                # Fallback: use regular numpy operations if numba not available
                for motif_sorted, int_motifs in zip(valid_motifs_sorted, valid_motifs_data):
                    has_motif = np.all(neighbor_counts[:, int_motifs] > 0, axis=1)
                    motif_count = np.sum(has_motif)
                    support = motif_count / num_center_cells
                    motif_supports[motif_sorted][dataset_name].append(support)

            # Neighbor matrix will be garbage collected after this iteration

    # Now perform statistical tests for each motif
    results = []

    for motif_sorted, support_by_dataset in motif_supports.items():
        # Perform Mann-Whitney U test
        group1 = np.array(support_by_dataset[datasets_list[0]])
        group2 = np.array(support_by_dataset[datasets_list[1]])

        # Calculate means
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)

        # Perform statistical test
        if len(group1) > 0 and len(group2) > 0:
            stat, p_value = stats.mannwhitneyu(group1, group2,
                                               alternative='two-sided',
                                               method='auto')

            # Determine which dataset has higher frequency using rank median
            support_rank = pd.concat([pd.DataFrame(group1), pd.DataFrame(group2)]).rank()
            median_rank1 = support_rank[:len(group1)].median()[0]
            median_rank2 = support_rank[len(group1):].median()[0]

            if median_rank1 > median_rank2:
                higher_dataset = datasets_list[0]
            else:
                higher_dataset = datasets_list[1]
        else:
            p_value = 1.0
            higher_dataset = "neither"

        results.append({
            'itemsets': motif_sorted,
            f'support_{datasets_list[0]}_mean': mean1,
            f'support_{datasets_list[1]}_mean': mean2,
            'p_value': p_value,
            'dataset_higher_frequency': higher_dataset,
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Apply FDR correction
    if len(results_df) > 1:
        if_rejected, corrected_p_values = mt.fdrcorrection(
            results_df['p_value'].values,
            alpha=0.05,
            method='poscorr'
        )
        results_df['adj_pvals'] = corrected_p_values
        results_df['if_significant'] = if_rejected
    else:
        results_df['adj_pvals'] = results_df['p_value']
        results_df['if_significant'] = results_df['p_value'] < 0.05

    # Split results by dataset - only keep significant ones
    dataset0_results = results_df[
        (results_df['dataset_higher_frequency'] == datasets_list[0]) &
        (results_df['if_significant'])
    ][['itemsets', 'adj_pvals']].copy()

    dataset1_results = results_df[
        (results_df['dataset_higher_frequency'] == datasets_list[1]) &
        (results_df['if_significant'])
    ][['itemsets', 'adj_pvals']].copy()

    # Sort by adjusted p-value
    dataset0_results = dataset0_results.sort_values(by='adj_pvals', ascending=True, ignore_index=True)
    dataset1_results = dataset1_results.sort_values(by='adj_pvals', ascending=True, ignore_index=True)

    return {datasets_list[0]: dataset0_results, datasets_list[1]: dataset1_results}


def differential_analysis_motif_dist(
    spatial_queries: List,
    datasets_list: List[str],
    ct: str,
    motifs: Union[str, List[str], List[List[str]]],
    max_dist: float = 20,
    min_size: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    Test whether user-specified motifs are differentially enriched across two datasets
    using radius-based neighborhood approach.

    Parameters
    ----------
    spatial_queries : List
        List of spatial_query objects from spatial_query_multi
    datasets_list : List[str]
        Two dataset names for differential analysis (length must be 2)
    ct : str
        Cell type of interest as center point
    motifs : Union[str, List[str], List[List[str]]]
        User-specified motif(s) to test. Can be:
        - Single cell type: 'CellTypeA'
        - Single motif: ['CellTypeA', 'CellTypeB']
        - Multiple motifs: [['CellTypeA'], ['CellTypeB', 'CellTypeC']]
    max_dist : float, default=20
        Maximum distance for considering a cell as a neighbor
    min_size : int, default=0
        Minimum neighborhood size for each point to consider

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys as dataset names, values as DataFrames with significant enriched motifs.
        Each DataFrame contains:
        - itemsets: the tested motif pattern (as tuple)
        - adj_pvals: FDR-corrected p-value
        Only significant motifs (adj_p_value < 0.05) for each dataset are included.
    """
    if len(datasets_list) != 2:
        raise ValueError("Require 2 datasets for differential analysis.")

    # Normalize motifs to list of lists
    if isinstance(motifs, str):
        # Single cell type as string: 'CellTypeA' -> [['CellTypeA']]
        motifs = [[motifs]]
    elif isinstance(motifs, list) and len(motifs) > 0 and isinstance(motifs[0], str):
        # Single motif as list of strings: ['CellTypeA', 'CellTypeB'] -> [['CellTypeA', 'CellTypeB']]
        motifs = [motifs]
    # else: already in correct format [['CellTypeA'], ['CellTypeB', 'CellTypeC']]

    # Initialize support storage for all motifs
    # Structure: {motif_tuple: {dataset_name: [support_fov1, support_fov2, ...]}}
    motif_supports = {}
    for motif in motifs:
        motif_sorted = tuple(sorted(motif))
        motif_supports[motif_sorted] = {datasets_list[0]: [], datasets_list[1]: []}

    # Process each FOV and compute support for all motifs
    for dataset_name in datasets_list:
        # Get all FOVs for this dataset
        dataset_queries = [sq for sq in spatial_queries
                         if sq.dataset.split('_')[0] == dataset_name]

        for sq in dataset_queries:
            cell_pos = sq.spatial_pos
            labels = sq.labels

            # Get center cell indices
            cinds = np.where(labels == ct)[0]
            ct_pos = cell_pos[cinds]
            num_center_cells = len(cinds)

            # Check if center cell type exists
            if num_center_cells == 0:
                for motif_sorted in motif_supports:
                    motif_supports[motif_sorted][dataset_name].append(0.0)
                continue


            # Vectorized computation: create neighbor count matrix
            label_encoder = LabelEncoder()
            int_labels = label_encoder.fit_transform(labels)

            num_types = len(label_encoder.classes_)

            # Query radius-based neighbors for center cells
            idxs = sq.kd_tree.query_ball_point(ct_pos, r=max_dist, return_sorted=False, workers=-1)

            # Filter out self from neighbors
            idxs_filter = [np.array(idx)[np.array(idx) != cind] for cind, idx in zip(cinds, idxs)]

            # Pre-compute neighbor matrix for center cells
            if len(idxs_filter) > 0 and any(len(idx) > 0 for idx in idxs_filter):
                flat_neighbors = np.concatenate([idx for idx in idxs_filter if len(idx) > 0])
                row_indices = np.repeat(np.arange(len(cinds)), [len(neigh) for neigh in idxs_filter])
                neighbor_labels = int_labels[flat_neighbors]

                neighbor_matrix = np.zeros((num_center_cells, num_types), dtype=int)
                np.add.at(neighbor_matrix, (row_indices, neighbor_labels), 1)
            else:
                neighbor_matrix = np.zeros((num_center_cells, num_types), dtype=int)

            # Compute neighborhood sizes
            neighborhood_sizes = np.array([len(idx) for idx in idxs_filter])

            # Now compute support for all motifs in this FOV - OPTIMIZED with Numba
            labels_unique = label_encoder.classes_
            labels_set = set(labels_unique)

            # Pre-compute: valid size mask (used by all motifs)
            has_valid_size = neighborhood_sizes > min_size

            # Separate valid and invalid motifs
            valid_motifs_sorted = []
            valid_motifs_data = []

            for motif in motifs:
                motif_sorted = tuple(sorted(motif))

                # Quick check: all cell types exist in this FOV?
                if all(m in labels_set for m in motif):
                    # Convert to integer indices
                    int_motifs = label_encoder.transform(np.array(motif))
                    valid_motifs_sorted.append(motif_sorted)
                    valid_motifs_data.append(int_motifs)
                else:
                    # Motif cell types don't exist in this FOV
                    motif_supports[motif_sorted][dataset_name].append(0.0)

            # If there are valid motifs, use numba-accelerated batch processing
            if len(valid_motifs_sorted) > 0 and NUMBA_AVAILABLE:
                # Prepare data for numba function
                max_motif_len = max(len(m) for m in valid_motifs_data)
                motif_indices_array = np.full((len(valid_motifs_data), max_motif_len), -1, dtype=np.int32)
                motif_lengths = np.array([len(m) for m in valid_motifs_data], dtype=np.int32)

                for i, int_motifs in enumerate(valid_motifs_data):
                    motif_indices_array[i, :len(int_motifs)] = int_motifs

                # Call numba-accelerated function
                supports = _compute_motif_support_numba(
                    neighbor_matrix.astype(np.int32),
                    motif_indices_array,
                    motif_lengths,
                    has_valid_size,
                    num_center_cells
                )

                # Store results
                for motif_sorted, support in zip(valid_motifs_sorted, supports):
                    motif_supports[motif_sorted][dataset_name].append(support)

            elif len(valid_motifs_sorted) > 0:
                # Fallback: use regular numpy operations if numba not available
                for motif_sorted, int_motifs in zip(valid_motifs_sorted, valid_motifs_data):
                    has_motif = np.all(neighbor_matrix[:, int_motifs] > 0, axis=1)
                    motif_count = np.sum(has_motif & has_valid_size)
                    support = motif_count / num_center_cells
                    motif_supports[motif_sorted][dataset_name].append(support)

            # Neighbor matrix will be garbage collected after this iteration

    # Now perform statistical tests for each motif
    results = []

    for motif_sorted, support_by_dataset in motif_supports.items():

        # Perform Mann-Whitney U test
        group1 = np.array(support_by_dataset[datasets_list[0]])
        group2 = np.array(support_by_dataset[datasets_list[1]])

        # Calculate means
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)

        # Perform statistical test
        if len(group1) > 0 and len(group2) > 0:
            stat, p_value = stats.mannwhitneyu(group1, group2,
                                               alternative='two-sided',
                                               method='auto')

            # Determine which dataset has higher frequency using rank median
            support_rank = pd.concat([pd.DataFrame(group1), pd.DataFrame(group2)]).rank()
            median_rank1 = support_rank[:len(group1)].median()[0]
            median_rank2 = support_rank[len(group1):].median()[0]

            if median_rank1 > median_rank2:
                higher_dataset = datasets_list[0]
            else:
                higher_dataset = datasets_list[1]
        else:
            p_value = 1.0
            higher_dataset = "neither"

        results.append({
            'itemsets': motif_sorted,
            f'support_{datasets_list[0]}_mean': mean1,
            f'support_{datasets_list[1]}_mean': mean2,
            'p_value': p_value,
            'dataset_higher_frequency': higher_dataset,
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Apply FDR correction
    if len(results_df) > 1:
        if_rejected, corrected_p_values = mt.fdrcorrection(
            results_df['p_value'].values,
            alpha=0.05,
            method='poscorr'
        )
        results_df['adj_pvals'] = corrected_p_values
        results_df['if_significant'] = if_rejected
    else:
        results_df['adj_pvals'] = results_df['p_value']
        results_df['if_significant'] = results_df['p_value'] < 0.05

    # Split results by dataset - only keep significant ones
    dataset0_results = results_df[
        (results_df['dataset_higher_frequency'] == datasets_list[0]) &
        (results_df['if_significant'])
    ][['itemsets', 'adj_pvals']].copy()

    dataset1_results = results_df[
        (results_df['dataset_higher_frequency'] == datasets_list[1]) &
        (results_df['if_significant'])
    ][['itemsets', 'adj_pvals']].copy()

    # Sort by adjusted p-value
    dataset0_results = dataset0_results.sort_values(by='adj_pvals', ascending=True, ignore_index=True)
    dataset1_results = dataset1_results.sort_values(by='adj_pvals', ascending=True, ignore_index=True)

    return {datasets_list[0]: dataset0_results, datasets_list[1]: dataset1_results}
