#include "LinearAlgebra/LinearSolvers.hpp"
#include "LinearAlgebra/SparseMatrix.hpp"
#include <cmath>

void Preconditioner::compute(const SparseMatrix &mat, PreconditionerType type) {
  p_type_ = type;
  diag_inv_.resize(0);
  L_.reset();
  U_.reset();

  switch (p_type_) {
  case PreconditionerType::None:
    // No preconditioning - nothing to compute
    break;

  case PreconditionerType::Diagonal: {
    // Compute inverse diagonal: M^-1 where M = diag(A)
    std::vector<double> diag =
        const_cast<SparseMatrix &>(mat).getDiagonal(mat.n);
    diag_inv_.resize(mat.n);
    for (size_t i = 0; i < mat.n; ++i) {
      diag_inv_[i] = (std::abs(diag[i]) > 1e-20) ? 1.0 / diag[i] : 1.0;
    }
    break;
  }

  case PreconditionerType::ILU0: {
    // Compute ILU(0) factorization: A ≈ L * U
    L_ = std::make_unique<SparseMatrix>(mat.n);
    U_ = std::make_unique<SparseMatrix>(mat.n);
    SparseMatrix::iluFactor(mat, *L_, *U_);
    break;
  }
  }
}

void Preconditioner::apply(std::vector<double> &r) const {
  assert(r.size() == diag_inv_.size());
  switch (p_type_) {
  case PreconditionerType::None:
    return;

  case PreconditionerType::Diagonal:
    // Diagonal preconditioning: z = D^-1 * r
    if (diag_inv_.size() > 0) {
      for (size_t i = 0; i < r.size(); i++) {
        r[i] *= diag_inv_[i];
      }
    }
    return;

  case PreconditionerType::ILU0:
    // ILU(0) preconditioning: solve L*U*z = r
    if (L_ && U_) {
      r = SparseMatrix::iluSolve(*L_, *U_, r);
    }
    return;
  }
}

double LinearSolver::solveCG(const size_t max_iter, const double tolerance,
                             const SparseMatrix &A,
                             const std::vector<double> &b,
                             std::vector<double> &x,
                             const Preconditioner &precond) {
  const size_t n = b.size();
  std::vector<double> a_by_x = A.multiply(x);
  std::vector<double> r(n);
  for (size_t i = 0; i < n; i++) {
    r[i] = b[i] - a_by_x[i];
  }

  // Apply preconditioner: z = M^-1 r
  std::vector<double> z = r;
  precond.apply(z);

  std::vector<double> p = z;
  double rsold = 0.0;
  for (size_t i = 0; i < n; ++i)
    rsold += r[i] * z[i];

  double rsnew = 0.0;
  for (size_t i = 0; i < max_iter; ++i) {
    std::vector<double> Ap = A.multiply(p);
    double pAp = 0.0;
    for (size_t k = 0; k < n; ++k)
      pAp += p[k] * Ap[k];

    if (std::abs(pAp) < 1e-20)
      break;
    double alpha = rsold / pAp;

    for (size_t k = 0; k < n; ++k) {
      x[k] += alpha * p[k];
      r[k] -= alpha * Ap[k];
    }

    // Apply preconditioner: z = M^-1 r
    z = r;
    precond.apply(z);

    rsnew = 0.0;
    for (size_t k = 0; k < n; ++k)
      rsnew += r[k] * z[k];

    if (std::sqrt(rsnew) < tolerance)
      return rsnew;

    for (size_t k = 0; k < n; ++k)
      p[k] = z[k] + (rsnew / rsold) * p[k];
    rsold = rsnew;
  }
  return rsnew;
}
double LinearSolver::solveBiCGSTAB(const size_t max_iter,
                                   const double tolerance,
                                   const SparseMatrix &A,
                                   const std::vector<double> &b,
                                   std::vector<double> &x,
                                   const Preconditioner &precond) {}
double LinearSolver::solveCG(const SparseMatrix &a,
                             const std::vector<double> &b,
                             std::vector<double> &x) {
  return LinearSolver::solveCG(n_corrections_, convergence_tolerance_, a, b, x,
                               preconditioner_);
}
double LinearSolver::solveBiCGSTAB(const SparseMatrix &a,
                                   const std::vector<double> &b,
                                   std::vector<double> &x) {
  return LinearSolver::solveBiCGSTAB(n_corrections_, convergence_tolerance_, a,
                                     b, x, preconditioner_);
}
