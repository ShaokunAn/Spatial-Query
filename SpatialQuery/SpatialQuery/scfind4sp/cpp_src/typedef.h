#pragma once
#include <vector>
#include <bitset>
#include <map>
#include <deque>
#include <unordered_map>
#include <iostream>

#include "const.h"


typedef std::pair<unsigned short, std::bitset<BITS> > BitSet32;
typedef std::vector<bool> BoolVec;
typedef int EliasFanoID;
typedef int CellTypeID;


using Item = std::string;
using Transaction = std::vector<Item>;
using Pattern = std::pair<std::set<Item>, uint64_t>;

typedef struct
{
  int gene;
  int cell_type;
  int index;
} IndexRecord;

class GeneMeta
{
public:
  int total_reads;
  GeneMeta();
  void merge(const GeneMeta& other);
};

class CellMeta
{
public:
  int reads;
  int features;
  int getReads() const
  {
    return reads;
  }

  int getFeatures() const
  {
    return features;
  }
  CellMeta();
};


class CellType
{
public:
  std::string name;
  int total_cells;
  int getTotalCells()const
  {
    return total_cells;
  }
};



typedef struct
{
  double mu;
  double sigma;
  std::vector<bool> quantile;
} Quantile;

class EliasFano
{
public:
  BoolVec H;
  BoolVec L;
  int l;
  float idf; // tfidf
  Quantile expr;
  int getSize() const
  {
    return L.size() / l;
  }
};


class CellID
{
public:
  CellTypeID cell_type;
  int cell_id;
  CellID(CellTypeID, int);
  bool operator==(const CellID& obj) const
  {
    return (obj.cell_type == cell_type) && (obj.cell_id == cell_id);

  }
  bool operator<(const CellID& other) const
  {
    if(other.cell_type != cell_type)
    {
      return other.cell_type > cell_type;
    }
    else
    {
      return other.cell_id > cell_id;
    }
  }


};


namespace std
{
  template<>
  struct hash<CellID>
  {
    inline size_t operator()(const CellID& cid) const
    {
      return hash<CellTypeID>()(cid.cell_type) ^ hash<int>()(cid.cell_id);
    }
  };

}

struct CellTypeMarker
{
  int tp;
  int fp;
  int tn;
  int fn;
  float inv_precision() const
  {
    return  (tp + fp) / float(tp);
  }
  float inv_recall() const
  {
    return (fn + tp) /  float(tp);
  }

  float recall() const
  {
    return 1 / inv_recall();
  }

  float precision() const
  {
    return 1 / inv_precision();
  }

  float f1() const
  {
    return 2/(inv_precision() + inv_recall());
  }
};

namespace arma {

// typedef unsigned int uword;

class vec {
public:
  vec(size_t size) : data(size, 0.0) {}

  double& operator()(size_t idx) { return data[idx]; }
  const double& operator()(size_t idx) const { return data[idx]; }
  double& operator[](size_t idx) { return data[idx]; }
  const double& operator[](size_t idx) const { return data[idx]; }

  size_t size() const { return data.size(); }

private:
  std::vector<double> data;
};

class sp_colvec {
public:
    sp_colvec(size_t nnz) : nnz(nnz) {}

    double get(size_t index) const {
        auto it = data.find(index);
        if (it != data.end()) {
            return it->second;
        }
        return 0.0;
    }

    void set(size_t index, double value) {
        if (value != 0) {
            data[index] = value;
        } 
        // else {
        //     data.erase(index);
        // }
    }

    class const_iterator {
    public:
        // const_iterator(typename std::unordered_map<uword, double>::const_iterator it) : it(it) {}
        const_iterator(typename std::map<size_t, double>::const_iterator it) : it(it) {}

        bool operator!=(const const_iterator& other) const { return it != other.it; }
        const_iterator& operator++() { ++it; return *this; }

        size_t row() const { return it->first; }
        double value() const { return it->second; }

    private:
        // typename std::unordered_map<uword, double>::const_iterator it;
        typename std::map<size_t, double>::const_iterator it;
    };

    const_iterator begin() const { return const_iterator(data.begin()); }
    const_iterator end() const { return const_iterator(data.end()); }

    size_t nnz;
    // std::unordered_map<uword, double> data;
    std::map<size_t, double> data;
};

class umat {
public:
    umat(size_t n_rows, size_t n_cols) : n_rows(n_rows), n_cols(n_cols), data(n_rows, std::vector<size_t>(n_cols, 0)) {}

    size_t& operator()(size_t row, size_t col) { return data[row][col]; }
    const size_t& operator()(size_t row, size_t col) const { return data[row][col]; }

    size_t getRows() const { return n_rows; }
    size_t getCols() const { return n_cols; }

    size_t n_rows;
    size_t n_cols;
    std::vector<std::vector<size_t>> data;
};

class sp_mat {
public:
    sp_mat(const umat& locations, const vec& values){
        size_t nnz = values.size();
        if (locations.getRows() != 2 || locations.getCols() != nnz) {
            throw std::invalid_argument("Dimensions of locations do not match number of non-zero elements");
        }

        // Find the maximum row and column indices to determine the matrix dimensions
        size_t max_row = 0, max_col = 0;
        for (size_t k = 0; k < nnz; ++k) {
            if (locations(0, k) > max_row) {
                max_row = locations(0, k);
            }
            if (locations(1, k) > max_col) {
                max_col = locations(1, k);
            }
        }
        n_rows = max_row + 1;
        n_cols = max_col + 1;

        // Resize col_ptrs to have max_col + 2 elements (one extra for the end pointer)
        col_ptrs.resize(max_col + 2, 0);

        // Count non-zero elements in each column
        for (size_t k = 0; k < nnz; ++k) {
            size_t col = locations(1, k);
            if (col >= max_col + 1) {
                throw std::out_of_range("Column index out of bounds");
            }
            col_ptrs[col + 1]++;
        }

        // Cumulative sum to get column pointers
        for (size_t col = 0; col < max_col + 1; ++col) {
            col_ptrs[col + 1] += col_ptrs[col];
        }

        // Allocate space for row indices and values
        row_indices.resize(nnz);
        values_.resize(nnz);

        // Fill row indices and values
        std::vector<size_t> col_count(max_col + 1, 0);
        for (size_t k = 0; k < nnz; ++k) {
            size_t row = locations(0, k);
            size_t col = locations(1, k);
            double value = values(k);

            if (col >= max_col + 1) {
                throw std::out_of_range("Column index out of bounds");
            }
            if (row >= max_row + 1) {
                // std::cout<<"row="<<row<<", max_row="<<max_row<<std::endl;
                throw std::out_of_range("Row index out of bounds");
            }

            size_t index = col_ptrs[col] + col_count[col];
            if (index >= col_ptrs[col + 1]) {
                throw std::out_of_range("Index out of bounds while filling row indices and values");
            }
            row_indices[index] = row;
            values_[index] = value;
            col_count[col]++;
        }
    }

    double get(size_t row, size_t col) const {
        if (col >= col_ptrs.size() - 1) {
            std::cerr << "Column index out of bounds: " << col << std::endl;
            return 0.0;
        }
        for (size_t i = col_ptrs[col]; i < col_ptrs[col + 1]; ++i) {
            if (row_indices[i] == row) {
                return values_[i];
            }
        }
        return 0.0;
    }

    void set(size_t row, size_t col, double value) {
        if (col >= col_ptrs.size() - 1) {
              std::cerr << "Column index out of bounds: " << col << std::endl;
              return;
          }
        for (size_t i = col_ptrs[col]; i < col_ptrs[col + 1]; ++i) {
            if (row_indices[i] == row) {
                if (value == 0) {
                    // Shift left to remove element
                    for (size_t j = i; j < col_ptrs[col + 1] - 1; ++j) {
                        row_indices[j] = row_indices[j + 1];
                        values_[j] = values_[j + 1];
                    }
                    col_ptrs[col + 1]--;
                } else {
                    values_[i] = value;
                }
                return;
            }
        }

        if (value != 0) {
            
            // Insert new element
            size_t pos = col_ptrs[col + 1]++;
            if (pos >= row_indices.size()) {
                  std::cerr << "Position out of bounds: " << pos << std::endl;
                  return;
              }
            for (size_t j = col_ptrs.size() - 1; j > col + 1; --j) {
                col_ptrs[j]++;
            }
            row_indices.insert(row_indices.begin() + pos, row);
            values_.insert(values_.begin() + pos, value);
        }
    }

    sp_colvec col(size_t col_index) const {
        if (col_index >= col_ptrs.size() - 1) {
              std::cerr << "Column index out of bounds: " << col_index << std::endl;
              return sp_colvec(col_ptrs.size() - 1);
          }
        size_t colvec_nnz_size = col_ptrs[col_index + 1] - col_ptrs[col_index];
        // sp_colvec column(col_ptrs.size() - 1);
        sp_colvec column(colvec_nnz_size);
        for (size_t i = col_ptrs[col_index]; i < col_ptrs[col_index + 1]; ++i) {
            column.set(row_indices[i], values_[i]);
        }
        return column;
    }
    
    size_t n_rows;
    size_t n_cols;
    std::vector<size_t> col_ptrs;
    std::vector<size_t> row_indices;
    std::vector<double> values_;
};


class sp_rowvec {
public:
    sp_rowvec(size_t size) : size(size) {}

    double get(size_t index) const {
        auto it = data.find(index);
        if (it != data.end()) {
            return it->second;
        }
        return 0.0;
    }

    class const_iterator {
    public:
        const_iterator(typename std::unordered_map<size_t, double>::const_iterator it) : it(it) {}

        bool operator!=(const const_iterator& other) const { return it != other.it; }
        const_iterator& operator++() { ++it; return *this; }

        size_t col() const { return it->first; }
        double value() const { return it->second; }

    private:
        typename std::unordered_map<size_t, double>::const_iterator it;
    };

    const_iterator begin() const { return const_iterator(data.begin()); }
    const_iterator end() const { return const_iterator(data.end()); }

    size_t size;
    std::unordered_map<size_t, double> data;
};
}



