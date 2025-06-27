
#include <cmath>
#include <vector>
#include "utils.h"
#include <pybind11/stl.h>
#include "typedef.h"
#include "fp_growth.h"

std::string str_join( const std::vector<std::string>& elements, const char* const separator)
{
  switch (elements.size())
  {
    case 0:
      return "";
    case 1:
      return *(elements.begin());
    default:
      std::ostringstream os;
      std::copy(elements.cbegin(), elements.cend() - 1, std::ostream_iterator<std::string>(os, separator));
      os << *elements.crbegin();
      return os.str();
  }
}



// Accepts a vector, transforms and returns a quantization logical vector
// This function aims for space efficiency of the expression vector
Quantile lognormalcdf(const std::vector<int>& ids, const std::vector<double>& v_array, unsigned int bits, bool raw_counts) {
//    py::buffer_info v_info = v_array.request();
//    double* v_ptr = static_cast<double*>(v_info.ptr);

    std::function<double(const double&)> expr_tran = raw_counts ? [](const double& x) {return std::log(x + 1);}: [](const double& x){return x;};

    Quantile expr;
    expr.mu = std::accumulate(ids.begin(), ids.end(), 0.0, [&v_array, &expr_tran](const double& mean, const int& index){
        return mean + expr_tran(v_array[index - 1]);
    }) / ids.size();

    expr.sigma = std::sqrt(
        std::accumulate(
            ids.begin(),
            ids.end(),
            0.0,
            [&v_array, &expr, &expr_tran](const double& variance, const int& index){
                return std::pow(expr.mu - expr_tran(v_array[index - 1]), 2);
            }) / ids.size());

    expr.quantile.resize(ids.size() * bits, 0);
    int expr_quantile_i = 0;
    for (auto const& s : ids) {
        unsigned int t = std::round(normalCDF(expr_tran(v_array[s]), expr.mu, expr.sigma) * (1 << bits));
        std::bitset<BITS> q = int2bin_core(t);
        for (unsigned int i = 0; i < bits; ++i) {
            expr.quantile[expr_quantile_i++] = q[i];
        }
    }
    return expr;
}

float inverf(float x)
{
  float tt1, tt2, lnx, sgn;
  sgn = (x < 0) ? -1.0f : 1.0f;

  x = (1 - x)*(1 + x);        // x = 1 - x*x;uo
  lnx = logf(x);

  tt1 = 2 / (M_PI * 0.147) + 0.5f * lnx;
  tt2 = 1 / 0.147 * lnx;

  return(sgn*sqrtf(-tt1 + sqrtf(tt1*tt1 - tt2)));
}


double lognormalinv(const double& p, const double& mu, const double& sigma)
{
  return exp((inverf(2*p - 1) * sqrt(2) * sigma) + mu);
}



std::vector<double> decompressValues(const Quantile& q, const unsigned char& quantization_bits)
{
  int vector_size = q.quantile.size() / quantization_bits;
  std::vector<double> result(vector_size,0);

  if(quantization_bits > 16)
  {
    std::cerr << "Too much depth in the quantization bits!" << std::endl;
  }
  std::vector<double> bins((1 << quantization_bits));
  double bins_size = bins.size();
  // min value

  for(int i = 0; i < bins_size; ++i)
  {
    double cdf = (i + 0.5) / bins_size;
    bins[i] = lognormalinv(cdf, q.mu, q.sigma);

  }

  for (size_t i = 0; i < result.size(); ++i)
  {
    int quantile = 0;
    for (size_t j = 0; j < quantization_bits; ++j)
    {
      quantile |= (1 << q.quantile[(quantization_bits * i) + j]);
    }
    result[i] = bins[quantile];
  }
  return result;
}




int byteToBoolVector(const std::vector<char> buf, std::vector<bool>& bool_vec)
{
  bool_vec.resize((buf.size() << 3), 0);
  int c = 0;
  for (auto const& b : buf)
  {
    for ( int i = 0; i < 8; i++)
    {
      bool_vec[c++] = ((b >> i) & 1);
    }
  }

  return 0;

}


int getSizeBoolVector(const std::vector<bool>& v)
{
  int size = v.size() / 8;
  if (v.size() % 8 != 0)
  {
    ++size;
  }
  return size;
}


std::map<EliasFanoDB::CellTypeName, std::map<int, Transaction> > transposeResultToCell(const py::dict& genes_results) {

    std::map<EliasFanoDB::CellTypeName, std::map<int, Transaction> > cells;

    std::vector<std::string> gene_names;
    for (auto item : genes_results) {
        gene_names.push_back(item.first.cast<std::string>());
    }

    for (const auto& gene_name : gene_names) {
        const py::dict& gene_hits = genes_results[py::str(gene_name)].cast<py::dict>();
        for (const auto& item : gene_hits) {
            const std::string ct = py::str(item.first);

            if (cells.find(ct) == cells.end()) {
                cells[ct] = std::map<int, Transaction>();
            }

            std::vector<unsigned int> ids = item.second.cast<std::vector<unsigned int>>();
            auto& cell_index = cells[ct];
            for (const auto& id : ids) {
                if (cell_index.find(id) == cell_index.end()) {
                    cell_index[id] = Transaction();
                }
                cell_index[id].push_back(gene_name);
            }
        }
    }

    return cells;
}

// Wrapper function
std::set<Pattern> FPGrowthFrequentItemsetMining(const py::dict& genes_results, const unsigned int min_support_cutoff)
{
    const std::map<EliasFanoDB::CellTypeName, std::map<int, Transaction>> result_by_celltype = transposeResultToCell(genes_results);

    // Find out how many cells are in the query
    const unsigned int cells_present = std::accumulate(result_by_celltype.begin(),
                                                       result_by_celltype.end(),
                                                       0,
                                                       [](const int& sum, const std::pair<EliasFanoDB::CellTypeName, std::map<int, Transaction>>& celltype) {
                                                           return sum + celltype.second.size();
                                                       });

    std::cerr << "Query Results Transposed: found " << cells_present << " sets" << std::endl;

    // Collect all transactions for fp-growth
    std::vector<Transaction> transactions;
    transactions.reserve(cells_present);
    for (auto const& ct : result_by_celltype)
    {
        // Iterate Cells of cell type
        for (auto const& cl : ct.second)
        {
            if (cl.second.size() != 1)
            {
                transactions.push_back(std::vector<Item>(cl.second.begin(), cl.second.end()));
            }
        }
    }

    std::cerr << transactions.size() << " transactions" << std::endl;

    const FPTree fptree{transactions, min_support_cutoff};
    return fptree_growth(fptree);
}

class CellIDs
{
 public:
  EliasFanoDB::GeneName gene_name;
  std::deque<CellID> cell_ids;
  CellIDs(const EliasFanoDB::GeneName& gene_name) : gene_name(gene_name)
  {

  }
};

typedef std::vector< CellIDs > CellVectors;

int findAllGeneSets(const CellVectors& query_results, std::set<Pattern>& gene_sets, const unsigned int min_support_cutoff) {

  int gene_number = query_results.size();
  unsigned long gene_limit = 1 << (gene_number + 1) ;

  if(gene_number > 32)
  {
    return 1;
  }

  std::cerr << "Starting Exhaustive search with " << gene_number << " genes" << std::endl;


  // mask is a set of genes that each bit set states the gene presence
  for (unsigned long mask = 1; mask < gene_limit; ++mask) {
    // auto& vector it->second;
    std::deque<CellID> current_cell_vector;
    bool set = false;
    bool fail = false;
    Pattern current_pattern;
    // https://stackoverflow.com/a/7774362/4864088
    for(unsigned char i = 0; i < gene_number; ++i) {
      if (char(mask >> i) & (0x01)) {
        if (!set) {
          // on first gene set the set;
          current_cell_vector = query_results[i].cell_ids;
          current_pattern.first.insert(query_results[i].gene_name);
          current_pattern.second = current_cell_vector.size();
          set = true;
        } else {
            const auto& current_gene_cells = query_results[i].cell_ids;
            std::deque<CellID> intersection;
            std::set_intersection(
                current_gene_cells.begin(),
                current_gene_cells.end(),
                current_cell_vector.begin(),
                current_cell_vector.end(),
                std::back_inserter(intersection)
            );
          if (intersection.size() < min_support_cutoff) {
            // no reason to go further
            fail = true;
            break;
          } else {
              current_pattern.first.insert(query_results[i].gene_name);
              current_pattern.second = intersection.size();
              current_cell_vector = std::move(intersection);
          }
        }
      }
    }
    if (not fail)
    {
      gene_sets.insert(current_pattern);
    }
  }
  return 0;
}


std::set<Pattern> exhaustiveFrequentItemsetMining(const py::dict& genes_results, const unsigned int min_support_cutoff)
{

  std::set<Pattern> results;

  // Boiler Plate code
  std::cout << "In exhaustive... now." << std::endl;
  std::vector<std::string> gene_names;
  for (const auto& item : genes_results) {
    gene_names.push_back(item.first.cast<std::string>());
  }
  const std::vector<std::string> gene_names_const = gene_names;

  std::unordered_map<EliasFanoDB::CellTypeName,CellTypeID> celltype_ids;
  CellVectors gene_cell_vector;

  // Start inversing the list to a cell level
  for (auto const& gene_name : gene_names_const) {

    CellIDs cell_vector(gene_name);
    auto& cells = cell_vector.cell_ids;

    // Gene hits contains the cell type hierarchy
    const py::dict& gene_hits = genes_results[py::str(gene_name)].cast<py::dict>();
    for (const auto& item : gene_hits) {
      // Generate Cell Type ID
      std::string celltype_name = py::str(item.first);
      CellTypeID ct_id = celltype_ids.insert(
        std::make_pair(
          celltype_name,
          celltype_ids.size()
                       )).first->second;

      std::vector<unsigned int> ids  = item.second.cast<std::vector<unsigned int >>();
      for (const auto& id : ids)
      {
        cells.push_back(CellID(ct_id, id));
      }
    }

    // (Optimization) if genes have not a minimum support then remove them
    if (not (cell_vector.cell_ids.size() < min_support_cutoff))
    {
      gene_cell_vector.push_back(cell_vector);
    }

  }
  for(auto& cell: gene_cell_vector)
  {
    auto& cell_vector = cell.cell_ids;
    std::sort(cell_vector.begin(), cell_vector.end());
  }

  findAllGeneSets(gene_cell_vector, results, min_support_cutoff);
  std::cout << "Existing exh... now." << std::endl;

  return results;
}

double fisher_exact_test(int a, int b, int c, int d) {
    double n = a + b + c + d;
    auto log_factorial = [](int x) {
        double result = 0;
        for (int i = 2; i <= x; ++i) {
            result += std::log(i);
        }
        return result;
    };
    double p = log_factorial(a + b) + log_factorial(c + d) + log_factorial(a + c) + log_factorial(b + d)
              - log_factorial(a) - log_factorial(b) - log_factorial(c) - log_factorial(d) - log_factorial(n);
    return std::exp(p);
}



