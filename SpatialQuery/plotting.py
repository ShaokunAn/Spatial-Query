"""
Plotting functions for spatial_query.
This module contains visualization methods for spatial query analysis.
"""

from typing import List, Optional, Union
from unittest import skip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import cluster
import seaborn as sns
from matplotlib import cm
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import SpectralBiclustering

from . import spatial_utils


def plot_fov(sq_obj,
             min_cells_label: int = 50,
             title: str = 'Spatial distribution of cell types',
             figsize: tuple = (10, 5),
             save_path: Optional[str] = None):
    """
    Plot the cell type distribution of single fov.

    Parameter
    --------
    sq_obj : spatial_query object
        The spatial_query object
    min_cells_label:
        Minimum number of points in each cell type to display.
    title:
        Figure title.
    figsize:
        Figure size parameter.
    save_path:
        Path to save the figure.
        If None, the figure will not be saved.

    Return
    ------
    A figure.
    """
    cell_type_counts = sq_obj.labels.value_counts()
    n_colors = sum(cell_type_counts >= min_cells_label)
    colors = sns.color_palette('hsv', n_colors)

    color_counter = 0
    fig, ax = plt.subplots(figsize=figsize)

    # Iterate over each cell type
    for cell_type in sorted(sq_obj.labels.unique()):
        # Filter data for each cell type
        index = sq_obj.labels == cell_type
        index = np.where(index)[0]
        # Check if the cell type count is above the threshold
        if cell_type_counts[cell_type] >= min_cells_label:
            ax.scatter(sq_obj.spatial_pos[index, 0], sq_obj.spatial_pos[index, 1],
                       label=cell_type, color=colors[color_counter], s=1)
            color_counter += 1
        else:
            ax.scatter(sq_obj.spatial_pos[index, 0], sq_obj.spatial_pos[index, 1],
                       color='grey', s=1)

    handles, labels = ax.get_legend_handles_labels()

    # Modify labels to include count values
    new_labels = [f'{label} ({cell_type_counts[label]})' for label in labels]

    # Create new legend
    ax.legend(handles, new_labels, loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)

    plt.xlabel('Spatial X')
    plt.ylabel('Spatial Y')
    plt.title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    # Adjust layout to prevent clipping of ylabel and accommodate the legend
    plt.tight_layout(rect=[0, 0, 1.1, 1])
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_motif_grid(sq_obj,
                    motif: Union[str, List[str]],
                    figsize: tuple = (10, 5),
                    max_dist: float = 20,
                    save_path: Optional[str] = None):
    """
    Display the distribution of each motif around grid points.

    Parameter
    ---------
    sq_obj : spatial_query object
        The spatial_query object
    motif:
        Motif (names of cell types) to be colored
    max_dist:
        Spacing distance for building grid.
    figsize:
        Figure size.
    save_path:
        Path to save the figure.
        If None, the figure will not be saved.

    Return
    ------
    A figure.
    """
    if isinstance(motif, str):
        motif = [motif]

    labels_unique = sq_obj.labels.unique()
    motif_exc = [m for m in motif if m not in labels_unique]
    if len(motif_exc) != 0:
        print(f"Found no {motif_exc} in {sq_obj.label_key}. Ignoring them.")
    motif = [m for m in motif if m not in motif_exc]

    # Build mesh
    xmax, ymax = np.max(sq_obj.spatial_pos, axis=0)
    xmin, ymin = np.min(sq_obj.spatial_pos, axis=0)
    x_grid = np.arange(xmin - max_dist, xmax + max_dist, max_dist)
    y_grid = np.arange(ymin - max_dist, ymax + max_dist, max_dist)
    grid = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)

    idxs = sq_obj.kd_tree.query_ball_point(grid, r=max_dist, return_sorted=False, workers=-1)

    # Locate the index of grid points acting as centers with motif nearby
    id_center = []
    for i, idx in enumerate(idxs):
        ns = [sq_obj.labels[id] for id in idx]
        if spatial_utils.has_motif(neighbors=motif, labels=ns):
            id_center.append(i)

    # Locate the index of cell types contained in motif in the neighborhood of above grid points with motif nearby
    id_motif_celltype = set()
    for i in id_center:
        idx = idxs[i]
        for cell_id in idx:
            if sq_obj.labels[cell_id] in motif:
                id_motif_celltype.add(cell_id)

    # Plot above spots and center grid points
    # Set color map using motif cell types
    motif_unique = sorted(set(motif))
    n_colors = len(motif_unique)
    colors = sns.color_palette('hsv', n_colors)
    color_map = {ct: col for ct, col in zip(motif_unique, colors)}

    motif_spot_pos = sq_obj.spatial_pos[list(id_motif_celltype), :]
    motif_spot_label = sq_obj.labels[list(id_motif_celltype)]
    fig, ax = plt.subplots(figsize=figsize)

    # Plotting the grid lines
    for x in x_grid:
        ax.axvline(x, color='lightgray', linestyle='--', lw=0.5)

    for y in y_grid:
        ax.axhline(y, color='lightgray', linestyle='--', lw=0.5)

    ax.scatter(grid[id_center, 0], grid[id_center, 1], label='Grid Points',
               edgecolors='red', facecolors='none', s=8)

    # Plotting other spots as background
    bg_index = [i for i, _ in enumerate(sq_obj.labels) if i not in id_motif_celltype]
    bg_pos = sq_obj.spatial_pos[bg_index, :]
    ax.scatter(bg_pos[:, 0], bg_pos[:, 1], color='#D3D3D3', s=1)

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
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_motif_rand(sq_obj,
                    motif: Union[str, List[str]],
                    max_dist: float = 20,
                    n_points: int = 1000,
                    figsize: tuple = (10, 5),
                    seed: int = 2023,
                    save_path: Optional[str] = None):
    """
    Display the random sampled points with motif in radius-based neighborhood,
    and cell types of motif in the neighborhood of these random points.

    Parameter
    ---------
    sq_obj : spatial_query object
        The spatial_query object
    motif:
        Motif (names of cell types) to be colored
    max_dist:
        Radius for neighborhood search.
    n_points:
        Number of random points to generate.
    figsize:
        Figure size.
    seed:
        Set random seed for reproducible.
    save_path:
        Path to save the figure.
        If None, the figure will not be saved.

    Return
    ------
    A figure.
    """
    if isinstance(motif, str):
        motif = [motif]

    labels_unique = sq_obj.labels.unique()
    motif_exc = [m for m in motif if m not in labels_unique]
    if len(motif_exc) != 0:
        print(f"Found no {motif_exc} in {sq_obj.label_key}. Ignoring them.")
    motif = [m for m in motif if m not in motif_exc]

    # Random sample points
    xmax, ymax = np.max(sq_obj.spatial_pos, axis=0)
    xmin, ymin = np.min(sq_obj.spatial_pos, axis=0)
    np.random.seed(seed)
    pos = np.column_stack((np.random.rand(n_points) * (xmax - xmin) + xmin,
                           np.random.rand(n_points) * (ymax - ymin) + ymin))

    idxs = sq_obj.kd_tree.query_ball_point(pos, r=max_dist, return_sorted=False, workers=-1)

    # Locate the index of random points acting as centers with motif nearby
    id_center = []
    for i, idx in enumerate(idxs):
        ns = [sq_obj.labels[id] for id in idx]
        if spatial_utils.has_motif(neighbors=motif, labels=ns):
            id_center.append(i)

    # Locate the index of cell types contained in motif in the neighborhood of above random points with motif nearby
    id_motif_celltype = set()
    for i in id_center:
        idx = idxs[i]
        for cell_id in idx:
            if sq_obj.labels[cell_id] in motif:
                id_motif_celltype.add(cell_id)

    # Plot above spots and center random points
    # Set color map using motif cell types
    motif_unique = sorted(set(motif))
    n_colors = len(motif_unique)
    colors = sns.color_palette('hsv', n_colors)
    color_map = {ct: col for ct, col in zip(motif_unique, colors)}

    motif_spot_pos = sq_obj.spatial_pos[list(id_motif_celltype), :]
    motif_spot_label = sq_obj.labels[list(id_motif_celltype)]
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(pos[id_center, 0], pos[id_center, 1], label='Random Sampling Points',
               edgecolors='red', facecolors='none', s=8)

    # Plotting other spots as background
    bg_index = [i for i, _ in enumerate(sq_obj.labels) if i not in id_motif_celltype]
    bg_adata = sq_obj.spatial_pos[bg_index, :]
    ax.scatter(bg_adata[:, 0], bg_adata[:, 1], color='#D3D3D3', s=1)

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
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_motif_celltype(sq_obj,
                        ct: str,
                        motif: Union[str, List[str]],
                        max_dist: float = 20,
                        figsize: tuple = (5, 5),
                        save_path: Optional[str] = None):
    """
    Display the distribution of interested motifs in the radius-based neighborhood of certain cell type.
    This function is mainly used to visualize the results of motif_enrichment_dist. Make sure the input parameters
    are consistent with those of motif_enrichment_dist.

    Parameter
    ---------
    sq_obj : spatial_query object
        The spatial_query object
    ct:
        Cell type as the center cells.
    motif:
        Motif (names of cell types) to be colored.
    max_dist:
        Spacing distance for building grid. Make sure using the same value as that in find_patterns_grid.
    figsize:
        Figure size.
    save_path:
        Path to save the figure.
        If None, the figure will not be saved.

    Return
    ------
    A figure.
    """
    if isinstance(motif, str):
        motif = [motif]

    motif_exc = [m for m in motif if m not in sq_obj.labels.unique()]
    if len(motif_exc) != 0:
        print(f"Found no {motif_exc} in {sq_obj.label_key}. Ignoring them.")
    motif = [m for m in motif if m not in motif_exc]

    if ct not in sq_obj.labels.unique():
        raise ValueError(f"Found no {ct} in {sq_obj.label_key}!")

    cinds = [i for i, label in enumerate(sq_obj.labels) if label == ct]
    idxs = sq_obj.kd_tree.query_ball_point(sq_obj.spatial_pos, r=max_dist, return_sorted=False, workers=-1)

    # Find the index of cell type spots whose neighborhoods contain given motif
    label_encoder = LabelEncoder()
    int_labels = label_encoder.fit_transform(np.array(sq_obj.labels))
    int_ct = label_encoder.transform(np.array(ct, dtype=object, ndmin=1))
    int_motifs = label_encoder.transform(np.array(motif))

    num_cells = len(idxs)
    num_types = len(label_encoder.classes_)
    idxs_filter = [np.array(ids)[np.array(ids) != i] for i, ids in enumerate(idxs)]

    flat_neighbors = np.concatenate(idxs_filter)
    row_indices = np.repeat(np.arange(num_cells), [len(neigh) for neigh in idxs_filter])
    neighbor_labels = int_labels[flat_neighbors]

    neighbor_matrix = np.zeros((num_cells, num_types), dtype=int)
    np.add.at(neighbor_matrix, (row_indices, neighbor_labels), 1)

    mask = int_labels == int_ct
    inds = np.where(np.all(neighbor_matrix[mask][:, int_motifs] > 0, axis=1))[0]
    cind_with_motif = [cinds[i] for i in inds]

    # Locate the index of motifs in the neighborhood of center cell type
    motif_mask = np.isin(np.array(sq_obj.labels), motif)
    all_neighbors = np.concatenate(idxs[cind_with_motif])
    exclude_self_mask = ~np.isin(all_neighbors, cind_with_motif)
    valid_neighbors = all_neighbors[motif_mask[all_neighbors] & exclude_self_mask]
    id_motif_celltype = set(valid_neighbors)

    # Plot figures
    motif_unique = set(motif)
    n_colors = len(motif_unique)
    colors = sns.color_palette('hsv', n_colors)
    color_map = {ct: col for ct, col in zip(motif_unique, colors)}
    motif_spot_pos = sq_obj.spatial_pos[list(id_motif_celltype), :]
    motif_spot_label = sq_obj.labels[list(id_motif_celltype)]
    fig, ax = plt.subplots(figsize=figsize)

    # Plotting other spots as background
    labels_length = len(sq_obj.labels)
    id_motif_celltype_set = set(id_motif_celltype)
    cind_with_motif_set = set(cind_with_motif)
    bg_index = [i for i in range(labels_length) if i not in id_motif_celltype_set and i not in cind_with_motif_set]
    bg_adata = sq_obj.spatial_pos[bg_index, :]
    ax.scatter(bg_adata[:, 0], bg_adata[:, 1], color='#D3D3D3', s=1)

    # Plot center the cell type whose neighborhood contains motif
    ax.scatter(sq_obj.spatial_pos[cind_with_motif, 0],
               sq_obj.spatial_pos[cind_with_motif, 1],
               label=ct, edgecolors='red', facecolors='none', s=3)

    for ct_m in motif_unique:
        ct_ind = motif_spot_label == ct_m
        ax.scatter(motif_spot_pos[ct_ind, 0],
                   motif_spot_pos[ct_ind, 1],
                   label=ct_m, color=color_map[ct_m], s=1)

    ax.legend(title='motif', loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)
    plt.xlabel('Spatial X')
    plt.ylabel('Spatial Y')
    plt.title(f"Spatial distribution of motif around {ct}")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_all_center_motif(sq_obj,
                          ct: str,
                          ids: dict,
                          figsize: tuple = (6, 6),
                          save_path: Optional[str] = None):
    """
    Plot the cell type distribution of single fov.

    Parameter
    --------
    sq_obj : spatial_query object
        The spatial_query object
    ct:
        Center cell type.
    ids : dict
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
    figsize: tuple
        Figure size.
    save_path:
        Path to save the figure.
        If None, the figure will not be saved.

    Return
    ------
    A figure.
    """
    # Set labels for each group
    adata = sq_obj.adata.copy()
    adata.obs['tmp'] = 'other'

    center_with_motif = np.unique(ids['center_neighbor_motif_pair'][:, 0])
    center_without_motif = np.unique(ids['non_motif_center_neighbor_pair'][:, 0])
    neighbor_motif = np.unique(ids['center_neighbor_motif_pair'][:, 1])
    non_neighbor_motif = np.unique(ids['non-neighbor_motif_cells'])
    center_without_motif_neighbors = np.unique(ids['non_motif_center_neighbor_pair'][:, 1])

    adata.obs.iloc[center_with_motif, adata.obs.columns.get_loc('tmp')] = f'center {ct} with motif'
    adata.obs.iloc[center_without_motif, adata.obs.columns.get_loc('tmp')] = f'non-motif center {ct}'

    adata.obs.iloc[neighbor_motif, adata.obs.columns.get_loc('tmp')] = [f'neighbor motif: {sq_obj.labels[i]}' for i in neighbor_motif]
    adata.obs.iloc[non_neighbor_motif, adata.obs.columns.get_loc('tmp')] = [f'non-neighbor motif: {sq_obj.labels[i]}' for i in non_neighbor_motif]

    adata.obs.iloc[center_without_motif_neighbors, adata.obs.columns.get_loc('tmp')] = 'non-motif-center neighbors'

    neighbor_motif_types = adata.obs.iloc[neighbor_motif, adata.obs.columns.get_loc(sq_obj.label_key)].unique()
    non_neighbor_motif_types = adata.obs.iloc[non_neighbor_motif, adata.obs.columns.get_loc(sq_obj.label_key)].unique()

    color_dict = {
        'other': '#D3D3D3',  # light gray for other cells
        f'center {ct} with motif': "#9F0707",  # dark red
        f'non-motif center {ct}': "#075FB1",  # dark blue
        'non-motif-center neighbors': "#6DE7E9"  # light blue
    }

    # Set red colors for neighbor motif and purple colors for non-neighbor motif
    n_neighbor_types = len(neighbor_motif_types)
    if n_neighbor_types > 0:
        red_colors = cm.Reds(np.linspace(0.3, 0.6, n_neighbor_types))
        for i, cell_type in enumerate(neighbor_motif_types):
            color_dict[f'neighbor motif: {cell_type}'] = red_colors[i]

    # Set purple colors for non-neighbor motif
    n_non_neighbor_types = len(non_neighbor_motif_types)
    if n_non_neighbor_types > 0:
        purple_colors = cm.Greens(np.linspace(0.4, 0.8, n_non_neighbor_types))
        for i, cell_type in enumerate(non_neighbor_motif_types):
            color_dict[f'non-neighbor motif: {cell_type}'] = purple_colors[i]

    # Plot figure
    fig, ax = plt.subplots(figsize=figsize)
    sc.pl.embedding(adata, basis='X_spatial', color='tmp', palette=color_dict, size=10, ax=ax, show=False)
    ax.set_title(f'Cell types around {ct} with motif', fontsize=10)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_motif_enrichment_heatmap(enrich_df: pd.DataFrame,
                                   figsize: tuple = (7, 5),
                                   save_path: Optional[str] = None,
                                   title: Optional[str] = None,
                                   cmap: str = 'GnBu'):
    """
    Plot a heatmap showing the distribution of cell types in enriched motifs.

    Parameter
    ---------
    enrich_df : pd.DataFrame
        Output from motif_enrichment_dist or motif_enrichment_knn
    figsize : tuple
        Figure size, default is (7, 5)
    save_path : str, optional
        Path to save the figure. If None, the figure will not be saved.
    title : str, optional
        Figure title. If None, will use a default title based on center cell type.
    cmap : str
        Colormap for the heatmap, default is 'GnBu'

    Return
    ------
    A figure showing the heatmap of motif cell type distribution.
    """
    if len(enrich_df) == 0:
        print("No enriched motifs to plot.")
        return

    # Calculate frequency
    enrich = enrich_df.copy()
    enrich['frequency'] = enrich['n_center_motif'] / enrich['n_center']

    # Sort by frequency
    enrich = enrich.sort_values(by='frequency', ascending=True)

    # Create motif group labels
    enrich['motif_group'] = [f'motif_{i+1}' for i in range(len(enrich))]

    # Explode motifs list so each cell type becomes a row
    enrich_expanded = enrich.explode('motifs')

    # Create pivot table: rows are cell types, columns are motif groups
    heatmap_data = enrich_expanded.pivot_table(
        index='motifs',  # Rows: each cell type in motif
        columns='motif_group',  # Columns: each motif group
        values='frequency',
        aggfunc='first'
    )

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        linewidths=0.1,
        linecolor='lightgrey',
        annot=True,
        fmt='.3f',
        annot_kws={'fontsize': 12},
        cbar_kws={'label': 'Frequency'}
    )

    # Set title
    if title is None:
        if 'center' in enrich.columns and len(enrich['center'].unique()) == 1:
            ct = enrich['center'].iloc[0]
            title = f'Distribution of enriched motifs around {ct}'
        else:
            title = 'Distribution of enriched motifs'

    plt.title(title, fontsize=14, pad=20)
    plt.ylabel('', fontsize=12)
    plt.xlabel('Motifs', fontsize=12)
    plt.xticks(rotation=30, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    

def plot_differential_pattern_heatmap(
    diff_motif: dict,
    ct: str,
    figsize: Optional[tuple] = None,
    cmap: str = 'YlOrRd',
    save_path: Optional[str] = None
):
    """
    Plot a heatmap showing differential patterns from multiple datasets.

    This function visualizes the differential patterns returned by differential_analysis_dist,
    comparing motifs across different datasets (e.g., CLR vs DII). Each motif is represented
    as a column, with cell types as rows, and the heatmap colored by -log10(adjusted p-value).

    Parameters
    ----------
    diff_motif : dict
        Dictionary containing differential analysis results from differential_analysis_dist.
        Keys are dataset names (e.g., 'CLR', 'DII'), values are DataFrames with columns:
        'itemsets' (motifs), 'adj_pvals' (adjusted p-values)
    ct : str
        Center cell type name (used in the title)
    figsize : tuple, optional
        Figure size (width, height). If None, automatically calculated based on data dimensions
    cmap : str
        Colormap for the heatmap, default is 'YlOrRd'
    save_path : str, optional
        Path to save the figure. If None, the figure will not be saved.

    Returns
    -------
    None
        Displays and optionally saves the heatmap figure

    Example
    -------
    >>> diff_motif = sps.differential_analysis_dist(
    ...     ct='tumor cells',
    ...     datasets=['CLR', 'DII'],
    ...     max_dist=5,
    ...     min_support=0.01
    ... )
    >>> plot_differential_pattern_heatmap(
    ...     diff_motif=diff_motif,
    ...     ct='tumor cells',
    ...     save_path='differential_patterns.pdf'
    ... )
    """
    # Combine all datasets
    combined_dfs = []
    for dataset_name, df in diff_motif.items():
        df_copy = df.copy()
        df_copy['dataset'] = dataset_name
        df_copy['-log10(adj_pvals)'] = -np.log10(df_copy['adj_pvals'])
        combined_dfs.append(df_copy)

    diff_combined = pd.concat(combined_dfs, ignore_index=True)

    # Sort by dataset, then by -log10(adj_pvals)
    diff_combined = diff_combined.sort_values(
        by=['dataset', '-log10(adj_pvals)'],
        ascending=[True, True]
    ).reset_index(drop=True)

    # Create motif group labels for each dataset
    diff_combined['motif_num'] = diff_combined.groupby('dataset').cumcount() + 1
    diff_combined['motif_group'] = (diff_combined['dataset'] + '_motif_' +
                                    diff_combined['motif_num'].astype(str))

    # Explode itemsets (motifs)
    diff_expanded = diff_combined.explode('itemsets')

    # Create pivot table
    heatmap_data = diff_expanded.pivot_table(
        index='itemsets',
        columns='motif_group',
        values='-log10(adj_pvals)',
        aggfunc='first'
    )

    # Maintain column order
    col_order = diff_combined['motif_group'].tolist()
    heatmap_data = heatmap_data[col_order]

    # Calculate dataset boundaries for vertical lines
    datasets = diff_combined['dataset'].tolist()
    boundaries = [i for i in range(1, len(datasets)) if datasets[i] != datasets[i-1]]

    # Auto-calculate figure size if not provided
    if figsize is None:
        width = max(8, len(col_order) * 0.45)
        height = max(4, len(heatmap_data) * 0.45)
        figsize = (width, height)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        linewidths=0.5,
        linecolor='lightgrey',
        cbar_kws={'label': '-log10(adj p-value)'},
        ax=ax
    )

    # Draw boundaries between datasets
    for b in boundaries:
        ax.axvline(x=b, color='black', linewidth=2)

    # Format plot
    dataset_names = ' vs '.join(diff_combined['dataset'].unique())
    plt.title(f'Differential patterns around {ct} ({dataset_names})',
              fontsize=14, pad=20)
    plt.ylabel('Cell types', fontsize=12)
    plt.xlabel('Motifs', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def plot_gene_pair_heatmap(
    gene_pair_df: pd.DataFrame,
    figsize: tuple = (7, 5),
    save_path: Optional[str] = None,
):
    """
    Plot heatmaps for gene pairs with biclustering, separately for each cell type.

    Parameters
    ----------
    gene_pair_df : pd.DataFrame
        DataFrame containing gene pair information with columns:
        'cell_type', 'gene_center', 'gene_motif', 'combined_score', 'if_significant'
    figsize : tuple
        Figure size for each cell type subplot (width, height)
    save_path : str, optional
        Path to save the figure. If None, the figure will not be saved.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with cluster assignments for all cell types.
        Contains columns: 'cell_type', 'gene_center', 'gene_motif',
        'combined_score', 'cluster_type', 'cluster_row', 'cluster_col'
    """
    def draw_bicluster_on_axis(matrix, model, ax, title, cmap):
        # 排序
        row_order = np.argsort(model.row_labels_)
        col_order = np.argsort(model.column_labels_)
        mat = matrix.iloc[row_order, col_order]

        # cluster labels
        row_clusters = model.row_labels_[row_order]
        col_clusters = model.column_labels_[col_order]

        # cluster 边界
        row_boundaries = np.where(np.diff(row_clusters) != 0)[0] + 1
        col_boundaries = np.where(np.diff(col_clusters) != 0)[0] + 1

        # mask zero entries
        mask = (mat == 0)

        # 绘图
        sns.heatmap(np.log10(mat+1), cmap=cmap, mask=mask, ax=ax, cbar=True, linewidths=0.1, linecolor='lightgrey')
        cbar = ax.collections[0].colorbar   # 获取 heatmap 的 colorbar
        cbar.set_label("log10(score + 1)",
                    #    rotation=90,
                    fontsize=12)


        for rb in row_boundaries:
            ax.hlines(rb, xmin=0, xmax=mat.shape[1], colors='black', linewidth=1.5)

        for cb in col_boundaries:
            ax.vlines(cb, ymin=0, ymax=mat.shape[0], colors='black', linewidth=1.5)

        ax.set_title(title)
        ax.set_xlabel("Motif genes")
        ax.set_ylabel("Center genes")

    unique_ct = gene_pair_df['cell_type'].unique()
    fig, axes = plt.subplots(
        nrows=len(unique_ct), ncols=2,
        figsize=(figsize[0], figsize[1] * len(unique_ct)),
        squeeze=False
    )

    # Filter for significant pairs only
    gene_pair_df = gene_pair_df[gene_pair_df['if_significant']]

    # Store all cluster assignments for each cell type
    all_cluster_assignments = []

    for ct_idx, ct in enumerate(unique_ct):
        # Filter data for current cell type
        sub_gene_pair = gene_pair_df[gene_pair_df['cell_type'] == ct]

        df_pivot = sub_gene_pair.pivot_table(
            index='gene_center',
            columns='gene_motif',
            values='combined_score',
            fill_value=0
        )

        mask = sub_gene_pair.pivot_table(
            index='gene_center',
            columns='gene_motif',
            values='if_significant',
            fill_value=0
        ).astype(bool)

        df_pivot = df_pivot.where(mask, 0)

        # separate positive and negative pairs
        pos_mat = df_pivot.clip(lower=0)
        neg_mat = (-df_pivot.clip(upper=0))  # positive values 

        # delete empty rows and columns
        pos_mat = pos_mat.loc[(pos_mat.sum(axis=1) > 0), (pos_mat.sum(axis=0) > 0)]
        neg_mat = neg_mat.loc[(neg_mat.sum(axis=1) > 0), (neg_mat.sum(axis=0) > 0)]

        # if less than 2 rows or columns, skip
        skip_pos, skip_neg = False, False
        if pos_mat.shape[0] < 2 or pos_mat.shape[1] < 2:
            skip_pos = True
            print(f"Cell type {ct}: Skipping pos with {pos_mat.shape[0]} rows and {pos_mat.shape[1]} columns.")
        if neg_mat.shape[0] < 2 or neg_mat.shape[1] < 2:
            skip_neg = True
            print(f"Cell type {ct}: Skipping neg with {neg_mat.shape[0]} rows and {neg_mat.shape[1]} columns.")

        pos_pairs = pd.DataFrame()
        neg_pairs = pd.DataFrame()

        # biclustering for positive pairs
        if not skip_pos:
            model_pos = SpectralBiclustering(n_clusters=3, method='scale', random_state=42)
            model_pos.fit(pos_mat)
            ax_pos = axes[ct_idx, 0]  # Use ct_idx for row selection
            draw_bicluster_on_axis(
                pos_mat,
                model_pos,
                ax=ax_pos,
                title=f"{ct} - Positive biclusters",
                cmap="Reds"
            )
            # assign cluster labels
            pos_row_labels = pd.Series(model_pos.row_labels_, index=pos_mat.index)
            pos_col_labels = pd.Series(model_pos.column_labels_, index=pos_mat.columns)

            row_idx, col_idx = np.where(pos_mat.values > 0)

            # 向量化构建结果
            pos_pairs = pd.DataFrame({
                'gene_center': pos_mat.index[row_idx],
                'gene_motif': pos_mat.columns[col_idx],
                'combined_score': pos_mat.values[row_idx, col_idx],
                'cluster_type': 'positive',
                'cluster_row': pos_row_labels.iloc[row_idx].values,
                'cluster_col': pos_col_labels.iloc[col_idx].values
            })
        else:
            # Hide the axis if skipping
            axes[ct_idx, 0].axis('off')

        # biclustering for negative pairs
        if not skip_neg:
            model_neg = SpectralBiclustering(n_clusters=3, method='scale', random_state=42)
            model_neg.fit(neg_mat)

            ax_neg = axes[ct_idx, 1]  # Use ct_idx for row selection
            draw_bicluster_on_axis(
                neg_mat,
                model_neg,
                ax=ax_neg,
                title=f"{ct} - Negative biclusters",
                cmap="Blues"
            )
            neg_row_labels = pd.Series(model_neg.row_labels_, index=neg_mat.index)
            neg_col_labels = pd.Series(model_neg.column_labels_, index=neg_mat.columns)

            row_idx, col_idx = np.where(neg_mat.values > 0)

            neg_pairs = pd.DataFrame({
                'gene_center': neg_mat.index[row_idx],
                'gene_motif': neg_mat.columns[col_idx],
                'combined_score': -neg_mat.values[row_idx, col_idx],  #  negative values
                'cluster_type': 'negative',
                'cluster_row': neg_row_labels.iloc[row_idx].values,
                'cluster_col': neg_col_labels.iloc[col_idx].values
            })
        else:
            # Hide the axis if skipping
            axes[ct_idx, 1].axis('off')

        # Combine positive and negative pairs for this cell type
        if not pos_pairs.empty or not neg_pairs.empty:
            cluster_assignment = pd.concat([pos_pairs, neg_pairs], ignore_index=True)
            cluster_assignment['cell_type'] = ct
            all_cluster_assignments.append(cluster_assignment)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Concatenate all cluster assignments into a single DataFrame
    if all_cluster_assignments:
        return pd.concat(all_cluster_assignments, ignore_index=True)
    else:
        # Return empty DataFrame with expected columns if no results
        return pd.DataFrame(columns=['cell_type', 'gene_center', 'gene_motif',
                                     'combined_score', 'cluster_type', 'cluster_row', 'cluster_col'])


