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
    size_t n = mat.n;
    diag_inv_.resize(n);
    // Direct CSR access: Find diagonal elements without helper functions
    // This assumes the matrix is sorted or we just scan the row.
    for (size_t i = 0; i < n; ++i) {
      double diag_val = 1.0; // Default if missing (shouldn't happen in FEM)

      // Scan row 'i' to find column 'i'
      int start = mat.row_ptr[i];
      int end = mat.row_ptr[i + 1];

      for (int k = start; k < end; ++k) {
        if (mat.col_idx[k] == (int)i) {
          diag_val = mat.val[k];
          break;
        }
      }

      // Invert with safety check
      if (std::abs(diag_val) > 1e-15) {
        diag_inv_[i] = 1.0 / diag_val;
      } else {
        // If diagonal is zero (singular), use 1.0 to avoid NaN.
        // This usually indicates a boundary condition error.
        diag_inv_[i] = 1.0;
      }
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
  // If no preconditioner is selected, do nothing.
  if (p_type_ == PreconditionerType::None)
    return;

  if (p_type_ == PreconditionerType::Diagonal) {
    // Ensure sizes match; if they don't, skip preconditioning to avoid crash.
    if (diag_inv_.size() != r.size())
      return;
    for (size_t i = 0; i < r.size(); i++) {
      r[i] *= diag_inv_[i];
    }
    return;
  }

  if (p_type_ == PreconditionerType::ILU0) {
    if (L_ && U_ && (size_t)L_->n == r.size()) {
      r = SparseMatrix::iluSolve(*L_, *U_, r);
    }
    return;
  }
}

static void flatten(const std::vector<glm::dvec3> &input,
                    std::vector<double> &output) {
  output.resize(input.size() * 3);
  for (size_t i = 0; i < input.size(); ++i) {
    output[3 * i + 0] = input[i].x;
    output[3 * i + 1] = input[i].y;
    output[3 * i + 2] = input[i].z;
  }
}

static void unflatten(const std::vector<double> &input,
                      std::vector<glm::dvec3> &output) {
  // output should already be sized correctly
  for (size_t i = 0; i < output.size(); ++i) {
    output[i].x = input[3 * i + 0];
    output[i].y = input[3 * i + 1];
    output[i].z = input[3 * i + 2];
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
                                   const Preconditioner &precond) {
  const size_t n = b.size();

  // Initial Residual: r0 = b - A * x
  std::vector<double> Ax = A.multiply(x);
  std::vector<double> r(n);
  for (size_t i = 0; i < n; ++i) {
    r[i] = b[i] - Ax[i];
  }

  // r_tld (shadow residual) is usually chosen to be equal to r0 initially
  std::vector<double> r_tld = r;

  // Initialize workspace vectors
  std::vector<double> p(n, 0.0);
  std::vector<double> v(n, 0.0);
  std::vector<double> s(n, 0.0);
  std::vector<double> t(n, 0.0);
  std::vector<double> y(n, 0.0); // Preconditioned p
  std::vector<double> z(n, 0.0); // Preconditioned s

  double rho_prev = 1.0;
  double alpha = 1.0;
  double omega = 1.0;

  // Compute initial norm for convergence check
  double r_norm = 0.0;
  for (double val : r)
    r_norm += val * val;
  r_norm = std::sqrt(r_norm);

  if (r_norm < tolerance)
    return r_norm; // Already converged

  // Main Loop
  for (size_t iter = 0; iter < max_iter; ++iter) {

    // Rho = dot(r_tld, r)
    double rho = 0.0;
    for (size_t i = 0; i < n; ++i)
      rho += r_tld[i] * r[i];

    // Breakdown check: if rho is 0, the method fails
    if (std::abs(rho) < 1e-50) {
      // std::cerr << "BiCGSTAB Breakdown: rho too small." << std::endl;
      return r_norm;
    }

    if (iter > 0) {
      double beta = (rho / rho_prev) * (alpha / omega);

      // p = r + beta * (p - omega * v)
      for (size_t i = 0; i < n; ++i) {
        p[i] = r[i] + beta * (p[i] - omega * v[i]);
      }
    } else {
      // First iteration: p = r
      p = r;
    }

    // Preconditioning: y = M^-1 * p
    y = p;
    precond.apply(y);

    // v = A * y
    v = A.multiply(y);

    // Alpha = rho / dot(r_tld, v)
    double r_tld_dot_v = 0.0;
    for (size_t i = 0; i < n; ++i)
      r_tld_dot_v += r_tld[i] * v[i];

    if (std::abs(r_tld_dot_v) < 1e-50) {
      // std::cerr << "BiCGSTAB Breakdown: alpha denominator too small." <<
      // std::endl;
      return r_norm;
    }
    alpha = rho / r_tld_dot_v;

    // s = r - alpha * v
    for (size_t i = 0; i < n; ++i) {
      s[i] = r[i] - alpha * v[i];
    }

    // Early exit check on norm of s
    double s_norm = 0.0;
    for (double val : s)
      s_norm += val * val;
    if (std::sqrt(s_norm) < tolerance) {
      // Update x and exit: x = x + alpha * y
      for (size_t i = 0; i < n; ++i)
        x[i] += alpha * y[i];
      return std::sqrt(s_norm);
    }

    // Preconditioning: z = M^-1 * s
    z = s;
    precond.apply(z);

    //  t = A * z
    t = A.multiply(z);

    // Omega = dot(t, s) / dot(t, t)
    double t_dot_s = 0.0;
    double t_dot_t = 0.0;
    for (size_t i = 0; i < n; ++i) {
      t_dot_s += t[i] * s[i];
      t_dot_t += t[i] * t[i];
    }

    if (std::abs(t_dot_t) < 1e-50) {
      // std::cerr << "BiCGSTAB Breakdown: omega denominator too small." <<
      // std::endl; Update x as best effort and return
      for (size_t i = 0; i < n; ++i)
        x[i] += alpha * y[i];
      return std::sqrt(s_norm);
    }
    omega = t_dot_s / t_dot_t;

    // Update x and r
    // x = x + alpha * y + omega * z
    // r = s - omega * t
    r_norm = 0.0;
    for (size_t i = 0; i < n; ++i) {
      x[i] += alpha * y[i] + omega * z[i];
      r[i] = s[i] - omega * t[i];
      r_norm += r[i] * r[i];
    }
    r_norm = std::sqrt(r_norm);

    // Prepare for next iteration
    rho_prev = rho;

    if (r_norm < tolerance) {
      return r_norm;
    }

    // Breakdown check for omega
    if (std::abs(omega) < 1e-16) {
      // std::cerr << "BiCGSTAB Breakdown: omega too small." << std::endl;
      return r_norm;
    }
  }

  return r_norm;
}
double LinearSolver::solveCG(const SparseMatrix &a,
                             const std::vector<double> &b,
                             std::vector<double> &x) const {
  return LinearSolver::solveCG(n_corrections_, convergence_tolerance_, a, b, x,
                               preconditioner_);
}
double LinearSolver::solveBiCGSTAB(const SparseMatrix &a,
                                   const std::vector<double> &b,
                                   std::vector<double> &x) const {
  return LinearSolver::solveBiCGSTAB(n_corrections_, convergence_tolerance_, a,
                                     b, x, preconditioner_);
}

double LinearSolver::solveCG(const SparseMatrix &A,
                             const std::vector<glm::dvec3> &b,
                             std::vector<glm::dvec3> &x) const {

  // Reuse a member buffer 'temp_b_' and 'temp_x_' for memory optimization
  // local vectors are safer for threading.
  std::vector<double> b_flat;
  std::vector<double> x_flat;

  flatten(b, b_flat);
  flatten(x, x_flat); // Use current x as initial guess

  // Solve using the standard double implementation
  double residual = solveCG(n_corrections_, convergence_tolerance_, A, b_flat,
                            x_flat, preconditioner_);

  // Unflatten Result back to x
  unflatten(x_flat, x);

  return residual;
}

double LinearSolver::solveBiCGSTAB(const SparseMatrix &A,
                                   const std::vector<glm::dvec3> &b,
                                   std::vector<glm::dvec3> &x) const {
  std::vector<double> b_flat;
  std::vector<double> x_flat;

  flatten(b, b_flat);
  flatten(x, x_flat);

  double residual = solveBiCGSTAB(n_corrections_, convergence_tolerance_, A,
                                  b_flat, x_flat, preconditioner_);

  unflatten(x_flat, x);
  return residual;
}
