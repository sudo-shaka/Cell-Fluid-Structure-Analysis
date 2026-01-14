#include <LinearAlgebra/SparseMatrix.hpp>
#include <algorithm>
#include <cassert>
#include <tuple>
#include <unordered_map>
#include <vector>

/**
 * @brief Build a CSR (Compressed Sparse Row) representation from triplet
 * entries.
 *
 * This function constructs the sparse matrix M in CSR format from a list of
 * triplets (row, column, value). Duplicate (i,j) entries in the triplet list
 * are summed. Out-of-range triplets (indices < 0 or >= n) are ignored.
 *
 * Behavior and layout:
 * - M.n is set to n.
 * - M.row_ptr is resized to n+1 and filled so that row_ptr[i]..row_ptr[i+1]-1
 *   indexes the column/value entries belonging to row i.
 * - For each row i, all off-diagonal entries are collected and sorted by column
 * index in ascending order. The diagonal entry (column == i) is placed last
 * within the row.
 * - If a diagonal entry is missing for a row:
 *     - When ensure_positive_diag == true, a tiny positive value (1e-20) is
 * inserted on the diagonal to avoid exact-zero diagonals.
 *     - When ensure_positive_diag == false, a zero diagonal entry is inserted.
 * - If a diagonal entry exists but its absolute value is smaller than 1e-20 and
 *   ensure_positive_diag == true, the diagonal is adjusted to +/-1e-20
 * preserving sign.
 *
 * Memory and intermediate structures:
 * - Uses an intermediate vector<std::unordered_map<int,double>> of size n to
 *   accumulate and sum per-row entries before finalizing CSR arrays.
 * - M.col_idx and M.val are cleared before being filled.
 */
void SparseMatrix::buildCsrFromTriplets(
    size_t n, const std::vector<std::tuple<int, int, double>> &triplets,
    SparseMatrix &M, bool ensure_positive_diag) {
  M.n = n;
  M.row_ptr.assign(n + 1, 0);
  M.col_idx.clear();
  M.val.clear();
  // Accumulate entries per row with sum of duplicates
  std::vector<std::unordered_map<int, double>> rows(n);
  rows.shrink_to_fit();
  for (const auto &t : triplets) {
    int i = std::get<0>(t);
    int j = std::get<1>(t);
    double v = std::get<2>(t);
    if (i < 0 || j < 0 || i >= (int)n || j >= (int)n)
      continue;
    rows[i][j] += v;
  }

  M.row_ptr[0] = 0;
  for (size_t i = 0; i < n; ++i) {
    auto &mapRow = rows[i];
    double diagVal = 0.0;
    bool hasDiag = false;

    std::vector<std::pair<int, double>> off;
    off.reserve(mapRow.size());

    for (const auto &kv : mapRow) {
      if (kv.first == (int)i) {
        diagVal = kv.second;
        hasDiag = true;
      } else {
        off.emplace_back(kv.first, kv.second);
      }
    }

    std::sort(off.begin(), off.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    for (const auto &p : off) {
      M.col_idx.push_back(p.first);
      M.val.push_back(p.second);
    }

    if (!hasDiag) {
      diagVal = ensure_positive_diag ? 1e-20 : 0.0;
    } else if (ensure_positive_diag && std::abs(diagVal) < 1e-20) {
      diagVal = (diagVal >= 0.0 ? 1e-20 : -1e-20);
    }
    M.col_idx.push_back((int)i);
    M.val.push_back(diagVal);

    M.row_ptr[i + 1] = (int)M.col_idx.size();
  }
}

void SparseMatrix::buildRectangularCsr(
    const std::vector<std::tuple<int, int, double>> &triplets, size_t nrows,
    size_t ncols, SparseMatrix &M) {
  M.n = nrows;
  M.row_ptr.assign(nrows + 1, 0);
  M.col_idx.clear();
  M.val.clear();

  std::vector<std::unordered_map<int, double>> rows(nrows);
  for (const auto &t : triplets) {
    int i = std::get<0>(t);
    int j = std::get<1>(t);
    double v = std::get<2>(t);
    if (i < 0 || j < 0 || i >= (int)nrows || j >= (int)ncols)
      continue;
    rows[i][j] += v;
  }

  M.row_ptr[0] = 0;
  for (size_t i = 0; i < nrows; ++i) {
    auto &mapRow = rows[i];
    std::vector<std::pair<int, double>> sorted_entries(mapRow.begin(),
                                                       mapRow.end());
    std::sort(sorted_entries.begin(), sorted_entries.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    for (const auto &p : sorted_entries) {
      M.col_idx.push_back(p.first);
      M.val.push_back(p.second);
    }
    M.row_ptr[i + 1] = (int)M.col_idx.size();
  }
}

template <typename T>
std::vector<T> SparseMatrix::multiply(const std::vector<T> &x) const {
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

void SparseMatrix::iluFactor(const SparseMatrix &A, SparseMatrix &L,
                             SparseMatrix &U) {
  assert(A.n == L.n && A.n == U.n);
  size_t n = A.n;

  L.row_ptr = A.row_ptr;
  U.row_ptr = A.row_ptr;
  L.col_idx = A.col_idx;
  U.col_idx = A.col_idx;
  L.val.resize(A.val.size(), 0.0);
  U.val.resize(A.val.size(), 0.0);

  // Pre-cache diagonal positions for fast access
  std::vector<int> diag_pos(n);
  for (size_t i = 0; i < n; ++i) {
    diag_pos[i] = -1;
    for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; ++idx) {
      if (A.col_idx[idx] == (int)i) {
        diag_pos[i] = idx;
        break;
      }
    }
  }

  // Build fast column->index lookup for each row (only allocate what's needed)
  std::vector<std::unordered_map<int, int>> row_col_map(n);
  for (size_t i = 0; i < n; ++i) {
    for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; ++idx) {
      row_col_map[i][A.col_idx[idx]] = idx;
    }
  }

  // ILU(0) factorization with sparse accumulator for each row
  std::vector<double> w(n, 0.0); // Working array for current row
  std::vector<int> w_pattern;    // Track which elements are nonzero
  w_pattern.reserve(100);

  for (size_t i = 0; i < n; ++i) {
    // Initialize working array with A(i,:)
    w_pattern.clear();
    for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; ++idx) {
      int j = A.col_idx[idx];
      w[j] = A.val[idx];
      w_pattern.push_back(j);
    }

    // Eliminate entries below diagonal
    for (int j : w_pattern) {
      if (j >= (int)i)
        break; // Only process lower triangular part

      if (std::abs(w[j]) > 1e-20) {
        // Get U(j,j) from diagonal
        double u_jj = (diag_pos[j] >= 0) ? U.val[diag_pos[j]] : 1e-20;
        if (std::abs(u_jj) < 1e-20)
          u_jj = 1e-20;

        double l_ij = w[j] / u_jj;
        w[j] = l_ij; // Store L(i,j)

        // Update w -= l_ij * U(j,:)
        // Only update elements that exist in sparsity pattern
        for (int idx = U.row_ptr[j]; idx < U.row_ptr[j + 1]; ++idx) {
          int k = U.col_idx[idx];
          if (k > j && row_col_map[i].count(k)) {
            w[k] -= l_ij * U.val[idx];
          }
        }
      }
    }

    // Store L and U values back
    for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; ++idx) {
      int j = A.col_idx[idx];
      if (j < (int)i) {
        L.val[idx] = w[j];
      } else {
        U.val[idx] = w[j];
      }
      w[j] = 0.0; // Clear for next row
    }
  }

  // Cache U diagonal for fast back-substitution
  U.diag_cache.resize(n);
  for (size_t i = 0; i < n; ++i) {
    if (diag_pos[i] >= 0) {
      U.diag_cache[i] = U.val[diag_pos[i]];
      if (std::abs(U.diag_cache[i]) < 1e-20)
        U.diag_cache[i] = 1e-20;
    } else {
      U.diag_cache[i] = 1e-20;
    }
  }
}

std::vector<double> SparseMatrix::iluSolve(const SparseMatrix &L,
                                           const SparseMatrix &U,
                                           const std::vector<double> &b) {
  const size_t n = L.n;
  std::vector<double> y(n), x(n);

  const int *L_row = L.row_ptr.data();
  const int *L_col = L.col_idx.data();
  const double *L_val = L.val.data();

  const int *U_row = U.row_ptr.data();
  const int *U_col = U.col_idx.data();
  const double *U_val = U.val.data();

  const bool hasCache = (U.diag_cache.size() == n);

  // Forward solve: L * y = b
  for (size_t i = 0; i < n; ++i) {
    double sum = 0.0;
    const int rowStart = L_row[i];
    const int rowEnd = L_row[i + 1];
    for (int j = rowStart; j < rowEnd; ++j) {
      const int col = L_col[j];
      // Lower triangular part only (col < i)
      if (col < (int)i)
        sum += L_val[j] * y[col];
      else
        break; // relies on columns sorted in ascending order
    }
    y[i] = b[i] - sum;
  }

  // Backward solve: U * x = y
  for (int i = (int)n - 1; i >= 0; --i) {
    double sum = 0.0;
    const int rowStart = U_row[i];
    const int rowEnd = U_row[i + 1];
    for (int j = rowStart; j < rowEnd; ++j) {
      const int col = U_col[j];
      if (col > i)
        sum += U_val[j] * x[col];
    }
    const double diag_i = hasCache ? U.diag_cache[i] : U_val[U_row[i + 1] - 1];
    x[i] = (y[i] - sum) / (std::abs(diag_i) < 1e-20 ? 1e-20 : diag_i);
  }

  return x;
}

std::vector<double> SparseMatrix::iluSolve(const SparseMatrix &A,
                                           const std::vector<double> &b) {
  SparseMatrix L(A.n), U(A.n);
  iluFactor(A, L, U);
  return iluSolve(L, U, b);
}

std::vector<double> SparseMatrix::getDiagonal(size_t n) const {
  std::vector<double> diag(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    // Assume diagonal is the last entry in the row
    diag[i] = val[row_ptr[i + 1] - 1];
  }
  return diag;
}
