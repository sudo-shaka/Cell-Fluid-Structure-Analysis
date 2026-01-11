#pragma once

#include <glm/vec3.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

struct SparseMatrix {
  size_t n;                 // number of rows (= cols)
  std::vector<int> row_ptr; // size n+1
  std::vector<int> col_idx; // column indices
  std::vector<double> val;  // nonzero values
  // Optional cache for fast diagonal access (typically for U in ILU)
  std::vector<double> diag_cache; // size n when valid

  explicit SparseMatrix() : n(0) {}
  explicit SparseMatrix(size_t n_rows) : n(n_rows) {
    row_ptr.resize(n + 1, 0);
  };

  static void iluFactor(const SparseMatrix &A, SparseMatrix &L,
                        SparseMatrix &U);
  static std::vector<double> iluSolve(const SparseMatrix &L,
                                      const SparseMatrix &U,
                                      const std::vector<double> &b);
  static std::vector<double> iluSolve(const SparseMatrix &A,
                                      const std::vector<double> &b);
  static void buildCsrFromTriplets(
      size_t n, const std::vector<std::tuple<int, int, double>> &triplets,
      SparseMatrix &M, bool ensure_positive_diag = true);

  std::vector<double> multiply(const std::vector<double> &) const;
  std::vector<glm::dvec3> multiply(const std::vector<glm::dvec3> &) const;
  std::vector<double> getDiagonal(size_t n);

  inline void reserve(size_t nnz) {
    col_idx.reserve(nnz);
    val.reserve(nnz);
  }

  inline void clear() {
    n = 0;
    row_ptr.clear();
    col_idx.clear();
    val.clear();
  }

  inline void print_sample(size_t n_rows = 5) const {
    std::cout << "SparseMatrix sample (first " << n_rows << " rows):\n";
    for (size_t i = 0; i < std::min(n_rows, n); ++i) {
      for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
        std::cout << "  (" << i << "," << col_idx[j] << ") = " << std::setw(12)
                  << val[j] << "\n";
      }
    }
  }
};
