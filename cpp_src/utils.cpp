#include <set>
#include <iostream>
// #include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include "typedef.h"
namespace py = pybind11;

// py::array_t<ssize_t> findMaximalPatterns(py::array_t<py::array_t<int>> patterns) {
// py::array_t<ssize_t> findMaximalPatterns(py::list patterns) {
//     std::vector<Pattern> maximalPatterns{};

//     std::vector<std::vector<int>> cpp_itemsets;
    
//     for (auto pattern : patterns) {
//         py::array_t<int> py_pattern = pattern.cast<py::array_t<int>>();
//         std::vector<int> cpp_pattern(py_pattern.size());
//         std::memcpy(cpp_pattern.data(), py_pattern.data(), py_pattern.size() * sizeof(int));
//         cpp_itemsets.push_back(cpp_pattern);
//     }


//     std::vector<Itemset> maximalItemsets;
//     std::vector<ssize_t> maximalPatternIndices;

//     for (size_t i = 0; i < cpp_itemsets.size(); ++i) {
//         const auto& pattern = cpp_itemsets[i];
//         bool isMaximal = true;
//         std::vector<size_t> subsetsToRemove;

//         for (size_t j = 0; j < maximalItemsets.size(); ++j) {
//             const auto& maxPattern = maximalItemsets[j];
//             if (std::includes(pattern.begin(), pattern.end(), maxPattern.begin(), maxPattern.end())) {
//                 subsetsToRemove.push_back(j);
//             } else if (std::includes(maxPattern.begin(), maxPattern.end(), pattern.begin(), pattern.end())) {
//                 isMaximal = false;
//                 break;
//             }
//         }

//         for (size_t j = subsetsToRemove.size(); j > 0; --j) {
//             maximalItemsets.erase(maximalItemsets.begin() + subsetsToRemove[j - 1]);
//         }

//         if (isMaximal) {
//             maximalItemsets.push_back(pattern);
//             maximalPatternIndices.push_back(static_cast<ssize_t>(i));
//         }
//     }

//     ssize_t size = maximalPatternIndices.size();
//     py::array_t<ssize_t> result(size);
//     auto ptr_result = static_cast<ssize_t*>(result.request().ptr);
//     std::copy(maximalPatternIndices.begin(), maximalPatternIndices.end(), ptr_result);
//     return result;
// }

bool has_motif(const std::vector<std::string> &motif, const std::vector<std::string> &neighborLabels)
{
    std::unordered_map<std::string, int> freqMotif;
    for (const std::string &element : motif)
    {
        freqMotif[element]++;
    }
    for (const std::string &element : neighborLabels)
    {
        auto it = freqMotif.find(element);
        if (it != freqMotif.end())
        {
            it->second--;
            if (it->second == 0)
            {
                freqMotif.erase(it);
                if (freqMotif.empty())
                {
                    return true;
                }
            }
        }
    }
    return false;
}

std::pair<int, int> search_motif_knn(
    const std::vector<std::string> &motif,
    const py::array_t<int> &idxs,
    const py::array_t<double> &dists,
    const std::vector<std::string> &labels,
    const py::array_t<int> &cinds,
    float max_dist)
{
    auto dists_unchecked = dists.unchecked<2>();
    auto idxs_unchecked = idxs.unchecked<2>();
    auto cinds_unchecked = cinds.unchecked<1>();

    std::vector<std::string> sort_motif = motif;
    std::sort(sort_motif.begin(), sort_motif.end());

    int n_motif_ct = 0;
    for (ssize_t idx = 0; idx < cinds_unchecked.shape(0); ++idx) {
        int i = cinds_unchecked(idx);
        std::vector<int> inds;
        for (ssize_t j = 0; j < dists_unchecked.shape(1); ++j) {
            if (dists_unchecked(i, j) < max_dist) {
                inds.push_back(j);
            }
        }
        if (inds.size() > 1) {
            std::vector<std::string> neighbor_labels;
            for (size_t j = 1; j < inds.size(); ++j) {
                neighbor_labels.push_back(labels[idxs_unchecked(i, inds[j])]);
            }
            if (has_motif(sort_motif, neighbor_labels)) {
                n_motif_ct += 1;
            }
        }
    }

    int n_motif_labels = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        std::vector<std::string> neighbor_labels;
        for (ssize_t j = 1; j < idxs_unchecked.shape(1); ++j) {
            neighbor_labels.push_back(labels[idxs_unchecked(i, j)]);
        }
        if (has_motif(sort_motif, neighbor_labels)) {
            n_motif_labels += 1;
        }
    }

    return std::make_pair(n_motif_ct, n_motif_labels);
}

std::pair<int, int> search_motif_dist(
    const std::vector<std::string> &motif,
    const py::list idxs,
    const std::vector<std::string> &labels,
    const py::array_t<int> &cinds,
    int max_neighbors)
{
    auto cinds_unchecked = cinds.unchecked<1>();

    std::vector<std::vector<size_t>> idxs_vec;
    idxs_vec.reserve(idxs.size());
    for (const auto &outer_list : idxs) {
        py::list idx_list = outer_list.cast<py::list>();
        std::vector<size_t> inner_vec = idx_list.cast<std::vector<size_t>>();
        idxs_vec.push_back(std::move(inner_vec));
    }

    std::vector<std::string> sort_motif = motif;
    std::sort(sort_motif.begin(), sort_motif.end());

    int n_motif_ct = 0;
    for (ssize_t idx = 0; idx < cinds_unchecked.shape(0); ++idx) {
        size_t i = cinds_unchecked(idx);
        std::vector<int> inds;
        int e = std::min(static_cast<int>(idxs_vec[i].size()), max_neighbors);

        for (int j = 0; j < e; ++j) {
            if (idxs_vec[i][j] != i) {
                inds.push_back(idxs_vec[i][j]);
            }
        }
        if (inds.size() > 0) {
            std::vector<std::string> neighbor_labels;
            for (size_t j = 0; j < inds.size(); ++j) {
                neighbor_labels.push_back(labels[inds[j]]);
            }
            if (has_motif(sort_motif, neighbor_labels)) {
                n_motif_ct += 1;
            }
        }
    }

    int n_motif_labels = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        std::vector<int> inds;
        int e = std::min(static_cast<int>(idxs_vec[i].size()), max_neighbors);

        for (int j = 0; j < e; ++j) {
            if (idxs_vec[i][j] != i) {
                inds.push_back(idxs_vec[i][j]);
            }
        }
        if (inds.size() > 0){
            std::vector<std::string> neighbor_labels;
            for (size_t j = 0; j < inds.size(); ++j) {
                neighbor_labels.push_back(labels[inds[j]]);
            }
            if (has_motif(sort_motif, neighbor_labels)) {
                n_motif_labels += 1;
            }
        }
    }

    return std::make_pair(n_motif_ct, n_motif_labels);
}


PYBIND11_MODULE(spatial_module_utils, m) {
    m.doc() = "A module for utils functions of SpatialQuery.";
    m.def("search_motif_knn", &search_motif_knn, "Search for motif with kNN neighborhoods.");
    m.def("search_motif_dist", &search_motif_dist, "Search for motif with kNN neighborhoods.");
}
