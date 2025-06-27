#pragma once
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <utility>
#include <map>
#include <vector>
#include <set>
#include <unordered_map>

#include "const.h"
#include "typedef.h"

namespace py = pybind11;



//' @export EliasFanoDB
class EliasFanoDB
{
 public:

  typedef std::string GeneName;
  typedef std::unordered_map<CellTypeID, EliasFanoID> GeneContainer;
  typedef std::map<GeneName, GeneContainer> GeneExpressionDB;

  typedef std::map<GeneName, GeneMeta> GeneIndex;

  typedef std::unordered_map<CellID, CellMeta> CellIndex;

  typedef std::string CellTypeName;
  typedef std::unordered_map<CellTypeName, CellTypeID> CellTypeIndex;
  typedef std::deque<EliasFano> ExpressionMatrix;

 // private:

  GeneExpressionDB index;
  CellIndex cells;
  CellTypeIndex cell_types;

  std::deque<CellType> inverse_cell_type;

  GeneIndex genes;
  ExpressionMatrix ef_data;
  int warnings;
  bool issparse;
  unsigned int total_cells;
  unsigned char quantization_bits;


  EliasFanoDB();

  void dumpGenes() const;

  void clearDB();

  int setQuantizationBits(const unsigned int value);

  unsigned int getQuantizationBits() const;

  const EliasFano& getEntry(const GeneName& gene_name, const CellTypeName& cell_type) const;

  int loadByteStream(const py::bytes& stream);

  py::bytes getByteStream() const;

  long eliasFanoCoding(const std::vector<int>& ids, const std::vector<double> &values);

  std::vector<int> eliasFanoDecoding(const EliasFano& ef) const;

  int queryZeroGeneSupport(const py::list&) const;

  // This is invoked on slices of the expression matrix of the dataset
  long encodeMatrix(const std::string& cell_type_name, const py::object& csr_obj, const py::list& cell_type_genes);

  // Same as encodeMatrix but to handle dense matrix
  long encodeMatrix_dense(const std::string &cell_type_name, const py::array_t<double> &dense_mat, const py::list &cell_type_genes);


  py::list total_genes() const;

  // Get a vector that represents support for a set of genes with respect to a specific dataset
  py::dict totalCells(const py::list &genes, const py::list&) const;

  //
  py::list getGenesInDB() const;

  int getTotalCells(const py::list&) const;

  py::dict geneSupportInCellTypes(const py::list& gene_names, const py::list&) const;

  const CellType& getCellType(const CellTypeName& name ) const;

  const py::tuple getCellTypeMatrix(const CellTypeName& cell_type) const;

  int numberOfCellTypes(const py::list&) const;

  int cellsInDB() const;

  CellTypeIndex getCellTypeIDs(const std::set<std::string>& datasets) const;

  py::list getCellTypeSupport(py::list& cell_types);

  py::dict queryGenes(const py::list& gene_names, const py::list& datasets_active) const;

  size_t dataMemoryFootprint() const;

  size_t quantizationMemoryFootprint() const;

  size_t dbMemoryFootprint() const;

  // And query
  py::dict findCellTypes(const py::list& gene_names, const py::list& datasets_active) const;
  py::dict _findCellTypes(const std::vector<std::string>& gene_names, const std::vector<CellTypeName>& cell_types_bg) const;


  // TODO(Nikos) this function can be optimized.. It uses the native quering mechanism
  // that casts the results into native R data structures
  py::dict findMarkerGenes(const py::list& gene_list, py::list datasets_active, bool exhaustive = false, int user_cutoff = -1) const;


  py::dict _findCellTypeMarkers(const py::list& cell_types,
                                       const py::list& background,
                                       const std::vector<GeneName>&,
                                       int mode = ALL) const;


  py::dict findCellTypeMarkers(const py::list& cell_types,
                                      const py::list& background) const;

  py::dict evaluateCellTypeMarkers(const py::list& cell_types,
                                          const py::list& gene_set,
                                          const py::list& background);

  py::dict evaluateCellTypeMarkersAND(const py::list& cell_types,
                                          const py::list& gene_set,
                                          const py::list& background);




  // std::map<GeneName, CellTypeMarker> _cellTypeScore(const std::string& cell_type, const std::vector<std::string>& universe, const std::vector <GeneName>&, int mode = ALL) const;

  std::unordered_map<GeneName, std::tuple<int, int, int, int>> _cellTypeCount(const std::string &cell_type, const std::vector<std::string> &universe, const std::vector<GeneName> &gene_names, int mode = ALL) const;

  const std::set<std::string> _getValidCellTypes(std::vector<std::string> universe) const;


  const std::vector<CellTypeName> _getCellTypes() const;
  const std::vector<CellTypeName> _getCellTypes(const std::vector<std::string> &datasets) const;
//  const std::vector<CellTypeName> _getCellTypes(const py::list& datasets) const;

  const std::vector<CellTypeName> getCellTypes() const;

  py::dict getCellMeta(const std::string&, const int&) const;

  py::dict getCellTypeMeta(const std::string&) const;

  const arma::sp_mat csr_to_sp_mat(const py::object& csr_obj);
  int dbSize() const;

  void dumpEFsize(int) const;
  int sample(int index) const;

  std::vector<int> decode(int index) const;

  int insertNewCellType(const CellType& cell_type);

  int mergeDB(const EliasFanoDB& db);

  std::vector<py::dict> DEGenes(const py::list &ct1, const py::list &ct2, const py::list genes_obj, const double &min_fraction);
  std::vector<py::dict> DEGenesIndices(const py::list &indices1, const py::list &indices2, const py::list& genes_obj, double min_fraction);
  std::vector<py::dict> findCellExpressingGenesinIndices(const py::list &indices, const py::list& genes_obj);


};


