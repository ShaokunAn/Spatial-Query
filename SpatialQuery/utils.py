from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import LabelEncoder
import scanpy as sc


def maximal_patterns(fp,
                     key: str = 'itemsets',
                     ):
    """
    Retrieve the maximal patterns in provided frequent patterns.

    Parameter
    ---------
    fp:
        DataFrame with frequent patterns
    key:
        Column name representing the patterns.

    Return
    ------
    A dataframe with the maximal patterns.
    """
    itemsets = fp[key].apply(frozenset)

    # Find all subsets of each itemset
    subsets = set()
    for itemset in itemsets:
        for r in range(1, len(itemset)):
            subsets.update(frozenset(s) for s in combinations(itemset, r))

    # Identify maximal patterns (itemsets that are not subsets of any other)
    maximal_patterns = [tuple(sorted(itemset)) for itemset in itemsets if itemset not in subsets]
    # maximal_patterns_ = [list(p) for p in maximal_patterns]

    # Filter the original DataFrame to keep only the maximal patterns
    return fp[fp['itemsets'].isin(maximal_patterns)].reset_index(drop=True)


# TODO: query efficiency can be improved as in motif_enrichment_dist of single FOV, filtering cells
def retrieve_niche_pattern_freq(fp, sp, ct, max_dist):
    """
    Retrieve frequency of each cell type in frequent pattern (fp) around
    central cell type (ct) from single FOV (sp).

    Parameters:
    fp: List[str]
        list of cell types
    sp: spatial_query object
        spatial query object for single FOV
    ct: str
        central cell type
    max_dist: float
        radius of neighborhood

    Return:
        AnnData with central cell type and neighboring cells in fp
    """
    if ct not in sp.labels.unique():
        print(f"{ct} does not exist in FOV.")
        return pd.DataFrame(columns=fp)

    cinds = [i for i, label in enumerate(sp.labels) if label == ct]  # id of center cell type

    # ct_pos = self.spatial_pos[cinds]
    idxs = sp.kd_tree.query_ball_point(sp.spatial_pos, r=max_dist, return_sorted=False, workers=-1)

    label_encoder = LabelEncoder()
    int_labels = label_encoder.fit_transform(np.array(sp.labels))
    int_ct = label_encoder.transform(np.array(ct, dtype=object, ndmin=1))

    num_cells = len(idxs)
    num_types = len(label_encoder.classes_)
    idxs_filter = [np.array(ids)[np.array(ids) != i] for i, ids in enumerate(idxs)]
    flat_neighbors = np.concatenate(idxs_filter)
    row_indices = np.repeat(np.arange(num_cells), [len(neigh) for neigh in idxs_filter])
    neighbor_labels = int_labels[flat_neighbors]

    neighbor_matrix = np.zeros((num_cells, num_types), dtype=int)
    np.add.at(neighbor_matrix, (row_indices, neighbor_labels), 1)

    mask = int_labels == int_ct

    motif_exc = [m for m in fp if m not in sp.labels.unique()]
    if len(motif_exc) != 0:
        print(f"Found no {motif_exc} in {sp.label_key}. Ignoring them.")
    motif = [m for m in fp if m not in motif_exc]

    int_motifs = label_encoder.transform(np.array(fp))

    inds = np.where(np.all(neighbor_matrix[mask][:, int_motifs] > 0, axis=1))[0]
    cind_with_motif = [cinds[i] for i in inds]  # id of ct with fp nearby

    freqs = []
    for id in cind_with_motif:
        ns = idxs_filter[id]
        freq_all = Counter(sp.labels[ns])
        freq_fp = {ct: count / len(ns) for ct, count in freq_all.items() if ct in fp}
        freqs.append(freq_fp)

    return pd.DataFrame(freqs)

def plot_niche_pattern_freq(freqs):
    """
    Heatmap plot of frequency of patterns (cell type compositions) in niche.

    Parameter
    ---------
    freqs: Output of retrieve_niche_pattern_freq method.
    """

    for i, freq in freqs.items():
        freq['FOV'] = f"normal_{i}"
        freqs[i] = freq

    freq_fp_normal = pd.concat(freqs)
    freq_fp_normal.set_index(keys='FOV', drop=True, inplace=True)
    fp_data = ad.AnnData(X=freq_fp_normal)
    var = pd.DataFrame({'neighbor': freq_fp_normal.columns.tolist()})
    var.index = freq_fp_normal.columns.tolist()
    obs = pd.DataFrame({'FOV': freq_fp_normal.index.tolist()})
    obs.index = freq_fp_normal.index.tolist()
    fp_data.var = var
    fp_data.obs = obs

    sc.pl.heatmap(fp_data, var_names=fp_data.var_names, groupby='FOV', cmap='vlag')
