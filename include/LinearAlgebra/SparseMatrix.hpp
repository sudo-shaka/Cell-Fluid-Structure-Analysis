#pragma once

#include <glm/vec3.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

struct SparseMatrix {
  size_t n;                 // number of rows (= cols)
  size_t m;                 // number of columns (for rectangular matrices)
  std::vector<int> row_ptr; // size n+1
  std::vector<int> col_idx; // column indices
  std::vector<double> val;  // nonzero values
  // Optional cache for fast diagonal access (typically for U in ILU)
  std::vector<double> diag_cache; // size n when valid

  explicit SparseMatrix() : n(0), m(0) {}
  explicit SparseMatrix(size_t n_rows) : n(n_rows), m(n_rows) {
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
  static void
  buildRectangularCsr(const std::vector<std::tuple<int, int, double>> &triplets,
                      size_t n_rows, size_t n_cols, SparseMatrix &matrix);

  static void combineMatrices(const SparseMatrix &A, double scale_a,
                              const SparseMatrix &B, double scale_b,
                              SparseMatrix &result);

  template <typename T> std::vector<T> multiply(const std::vector<T> &x) const {
    std::vector<T> y(n);

    for (size_t i = 0; i < n; ++i) {
      T sum(0.0); // works for double and glm::dvec3
      for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
        sum += val[j] * x[col_idx[j]];
      }
      y[i] = sum;
    }
    return y;
  }

  template <typename T>
  std::vector<T> multiplyTranspose(const std::vector<T> &x) const {
    // Computes A^T * x where A is (n x m). x must have length n.
    assert(x.size() == n);
    const size_t n_cols = m; // may be equal to n for square matrices
    std::vector<T> result(n_cols, 0.0);

    for (size_t i = 0; i < n; ++i) {
      for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
        int col = col_idx[k];
        result[col] += val[k] * x[i];
      }
    }
    return result;
  }

  std::vector<double> getDiagonal(size_t n) const;

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

  inline void zeroRow(int row_idx) {
    for (int k = row_ptr[row_idx]; k < row_ptr[row_idx + 1]; ++k) {
      val[k] = 0.0;
    }
  }

  inline void set(size_t row, size_t col, double value) {
    assert(row < row_ptr.size() - 1);

    const int start = row_ptr[row];
    const int end = row_ptr[row + 1];

    for (int k = start; k < end; ++k) {
      if (col_idx[k] == (int)col) {
        val[k] = value;
        return;
      }
    }

    // If we get here, the entry does not exist in the sparsity pattern.
    // For FEM Boundary Conditions, the diagonal (i, i) should ALWAYS exist.
    std::cerr << "Error: Attempted to set(" << row << ", " << col
              << ") but entry does not exist in sparsity pattern.\n";
    throw std::runtime_error("Sparse matrix fill-in not supported");
  }

  // Returns the non-zero entries of a specific row as (column_index, value)
  // pairs.
  inline std::vector<std::pair<int, double>> getRow(size_t row_idx) const {
    assert(row_idx < n); // Safety check to ensure row exists

    std::vector<std::pair<int, double>> result;

    // In CSR format, the data for 'row_idx' is stored between
    // row_ptr[row_idx] and row_ptr[row_idx + 1]
    const int start = row_ptr[row_idx];
    const int end = row_ptr[row_idx + 1];

    // Reserve memory upfront to avoid multiple reallocations
    result.reserve(end - start);

    for (int k = start; k < end; ++k) {
      result.emplace_back(col_idx[k], val[k]);
    }

    return result;
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
