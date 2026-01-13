#pragma once

#include "BC/BC.hpp"
#include "LinearAlgebra/SparseMatrix.hpp"
#include <memory>
#include <vector>

enum class PreconditionerType { None, Diagonal, ILU0 };

class Preconditioner {
public:
  explicit Preconditioner() : p_type_(PreconditionerType::None) {}
  void compute(const SparseMatrix &mat, PreconditionerType type) const;
  std::vector<double> apply(const std::vector<double> &r) const;
  const PreconditionerType &getType() const { return p_type_; }
  bool isIntialized() const {
    return p_type_ != PreconditionerType::None || diag_inv_.size() > 0;
  }

private:
  PreconditionerType p_type_;
  std::vector<double> diag_inv_;    // for diagonal / jacabi preconditioner
  std::unique_ptr<SparseMatrix> L_; // for ILU0: lower triangular factor
  std::unique_ptr<SparseMatrix> U_; // for ILU0: for upper triangular factor
};

class LinearSolver {
  int n_corrections_;
  double convergence_tolerance_;
  Preconditioner preconditioner_;

public:
  static double solve_cg(const size_t max_iter, const double tolerance,
                         const SparseMatrix &A, const std::vector<double> &b,
                         std::vector<double> &x, const Preconditioner &precond);
  static double solve_bicstab(const size_t max_iter, const double tolerance,
                              const SparseMatrix &A,
                              const std::vector<double> &b,
                              std::vector<double> &x,
                              const Preconditioner &precond);
  double solve_cg(const SparseMatrix &a, const std::vector<double> &b,
                  std::vector<double> &x);
  double solve_bicgstab(const SparseMatrix &a, const std::vector<double> &b,
                        std::vector<double> &x);

  void setMaxCorrections(size_t max) { n_corrections_ = max; }
  void setTolerance(double tol) { convergence_tolerance_ = tol; }
  void setPreconditoner(Preconditioner precond) {
    preconditioner_ = std::move(precond);
  }
  PreconditionerType getPreconditionerType() const {
    return preconditioner_.getType();
  }
};
