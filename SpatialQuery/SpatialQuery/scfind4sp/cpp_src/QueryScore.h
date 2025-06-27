#pragma once

#include <vector>
#include <set>
#include <pybind11/pybind11.h>

#include "typedef.h"

namespace py = pybind11;

class EliasFanoDB;

class QueryScore
{
public:
  typedef struct
  {
    double tfidf;
    size_t index;
    int cartesian_product_sets;
    int support_in_datasets;
  } GeneScore;

  friend class EliasFanoDB;
  std::map<std::string, GeneScore> genes;
  std::unordered_map<CellID , std::pair<std::vector<double>, int> > tfidf;
  QueryScore();
  float cell_tfidf(const EliasFanoDB&, const std::set<std::string>&);
  void estimateExpression(const py::dict& gene_results, const EliasFanoDB& db, const py::list& datasets);
  unsigned int geneSetCutoffHeuristic(const float = 0.5);
  int calculate_cell_types(const std::set<std::string>&gene_set);
};
