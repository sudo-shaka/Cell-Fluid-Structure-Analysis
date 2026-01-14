#include "FEM/NavierStokes.hpp"
#include <array>
#include <cmath>
#include <tuple>
#include <vector>

// INTERNAL HELPER: Quadrature & Shape Functions
namespace FEM_Utils {

// 4-Point Gaussian Quadrature for Tetrahedron
// Exact for polynomials up to Degree 2.
// Coordinates are barycentric (lambda).
struct QuadPoint {
  double w;          // Weight
  glm::dvec4 lambda; // Barycentric coordinates (l0, l1, l2, l3)
};

// Derived from alpha = 0.58541020, beta = 0.13819660
static const double a = 0.5854101966249685;
static const double b = 0.1381966011250105;

static const std::array<QuadPoint, 4> quadrature_rule = {
    {{0.25, {a, b, b, b}},
     {0.25, {b, a, b, b}},
     {0.25, {b, b, a, b}},
     {0.25, {b, b, b, a}}}};

// Evaluate P1 Shape Functions (Linear)
// N = [l0, l1, l2, l3]
inline std::array<double, 4> eval_P1(const glm::dvec4 &l) {
  return {l[0], l[1], l[2], l[3]};
}

// Evaluate P2 Shape Functions (Quadratic)
// Order: Vertices (0-3), Edges (01, 02, 03, 12, 13, 23)
inline std::array<double, 10> eval_P2(const glm::dvec4 &l) {
  std::array<double, 10> N;
  // Vertices: l(2l - 1)
  for (int i = 0; i < 4; ++i)
    N[i] = l[i] * (2.0 * l[i] - 1.0);

  // Edges: 4 * l_i * l_j
  N[4] = 4.0 * l[0] * l[1]; // Edge 0-1
  N[5] = 4.0 * l[0] * l[2]; // Edge 0-2
  N[6] = 4.0 * l[0] * l[3]; // Edge 0-3
  N[7] = 4.0 * l[1] * l[2]; // Edge 1-2
  N[8] = 4.0 * l[1] * l[3]; // Edge 1-3
  N[9] = 4.0 * l[2] * l[3]; // Edge 2-3
  return N;
}

// Evaluate Gradient of P2 Shape Functions with respect to Lambda
// Returns 10x4 matrix (dN_i / dLambda_k)
inline std::array<glm::dvec4, 10> eval_grad_P2_wrt_lambda(const glm::dvec4 &l) {
  std::array<glm::dvec4, 10>
      dNdL; // 10 nodes, each has a vec4 gradient (w.r.t l0, l1, l2, l3)

  // Vertices: d/dl_k (l_k(2l_k - 1)) = 4l_k - 1 (diagonal only)
  for (int i = 0; i < 4; ++i) {
    dNdL[i] = glm::dvec4(0.0);
    dNdL[i][i] = 4.0 * l[i] - 1.0;
  }

  // Edges: d/dl (4*li*lj)
  // Gradient has entries at indices i and j
  dNdL[4] = glm::dvec4(4.0 * l[1], 4.0 * l[0], 0.0, 0.0); // 0-1
  dNdL[5] = glm::dvec4(4.0 * l[2], 0.0, 4.0 * l[0], 0.0); // 0-2
  dNdL[6] = glm::dvec4(4.0 * l[3], 0.0, 0.0, 4.0 * l[0]); // 0-3
  dNdL[7] = glm::dvec4(0.0, 4.0 * l[2], 4.0 * l[1], 0.0); // 1-2
  dNdL[8] = glm::dvec4(0.0, 4.0 * l[3], 0.0, 4.0 * l[1]); // 1-3
  dNdL[9] = glm::dvec4(0.0, 0.0, 4.0 * l[3], 4.0 * l[2]); // 2-3

  return dNdL;
}
} // namespace FEM_Utils


// SOLVER IMPLEMENTATION
void NavierStokesSolver::initialize(std::shared_ptr<Mesh> mesh_ptr,
                                    const Fluid &fluid_props) {
  mesh_ptr_ = mesh_ptr;
  is_initialized_ = true;

  const size_t nv = mesh_ptr_->nVertices();
  const size_t vert_dof =
      mesh_ptr_->getNumberOfEdgeNodes() + nv; // Total Velocity DOFs

  fluid_properties_ = fluid_props;
  fluid_properties_.effective_viscosity.resize(vert_dof, fluid_props.viscosity);

  velocity_.resize(vert_dof, glm::dvec3{0.0});
  velocity_star_.resize(vert_dof, glm::dvec3{0.0});
  pressure_.resize(nv, 0.0);
  pressure_correction_.resize(nv, 0.0);

  build_sparse_matrices();
}

// P2-P1 Gradient matrix: Maps P2 Velocity -> P1 Pressure
// Integral: phi_i^{P1} * div(N_j^{P2}) dVol
void NavierStokesSolver::buildGradientMatrices() {
  const size_t n_nodes = mesh_ptr_->nVertices();
  const size_t n_velocity_dof = mesh_ptr_->getP1plusP2DegreesOfFreedom();
  const size_t n_tets = mesh_ptr_->nTets();

  std::vector<std::tuple<int, int, double>> triplets_x, triplets_y, triplets_z;
  triplets_x.reserve(40 * n_tets);
  triplets_y.reserve(40 * n_tets);
  triplets_z.reserve(40 * n_tets);

  for (size_t ti = 0; ti < n_tets; ti++) {
    const auto &tet = mesh_ptr_->tetAt(ti);
    double vol = tet.volume;

    // Physical Gradients of Barycentric Coords (dL/dx, dL/dy, dL/dz)
    const auto &grad_lambda = mesh_ptr_->getTetGradient(ti);

    // Gather indices: 4 P1 nodes, 10 P2 nodes
    std::array<int, 4> p_idx; // Pressure (P1)
    for (int i = 0; i < 4; ++i)
      p_idx[i] = tet.vertids[i];

    std::array<int, 10> v_idx; // Velocity (P2)
    for (int i = 0; i < 4; ++i)
      v_idx[i] = tet.vertids[i]; // Vertices
    const auto &edge_nodes = mesh_ptr_->getTetEdgeNodes(ti);
    for (int i = 0; i < 6; ++i)
      v_idx[4 + i] = n_nodes + edge_nodes[i]; // Edges

    // --- Quadrature Loop ---
    for (const auto &qp : FEM_Utils::quadrature_rule) {
      // Evaluate P1 basis (Pressure)
      auto phi = FEM_Utils::eval_P1(qp.lambda);

      // Evaluate P2 gradients (Velocity)
      auto dNdL = FEM_Utils::eval_grad_P2_wrt_lambda(qp.lambda);

      // Convert dN/dLambda to dN/dX (Physical Gradient)
      // dN_j/dX = sum_k (dN_j/dL_k * dL_k/dX)
      std::array<glm::dvec3, 10> grad_N;
      for (int j = 0; j < 10; ++j) {
        grad_N[j] = glm::dvec3(0.0);
        for (int k = 0; k < 4; ++k) {
          grad_N[j] += dNdL[j][k] * grad_lambda[k];
        }
      }

      double weight = qp.w * vol; // Integration weight

      // Assembly: Integral of (Phi_i * dN_j/dx)
      for (int i = 0; i < 4; ++i) {    // Pressure Loop
        for (int j = 0; j < 10; ++j) { // Velocity Loop
          double factor = weight * phi[i];

          double val_x = factor * grad_N[j].x;
          double val_y = factor * grad_N[j].y;
          double val_z = factor * grad_N[j].z;

          if (std::abs(val_x) > 1e-16)
            triplets_x.emplace_back(p_idx[i], v_idx[j], val_x);
          if (std::abs(val_y) > 1e-16)
            triplets_y.emplace_back(p_idx[i], v_idx[j], val_y);
          if (std::abs(val_z) > 1e-16)
            triplets_z.emplace_back(p_idx[i], v_idx[j], val_z);
        }
      }
    }
  }

  gradient_matrix_x_ =
      std::make_unique<SparseMatrix>(n_nodes); // Note: Rectangular
  gradient_matrix_y_ = std::make_unique<SparseMatrix>(n_nodes);
  gradient_matrix_z_ = std::make_unique<SparseMatrix>(n_nodes);

  SparseMatrix::buildRectangularCsr(triplets_x, n_nodes, n_velocity_dof,
                                    *gradient_matrix_x_);
  SparseMatrix::buildRectangularCsr(triplets_y, n_nodes, n_velocity_dof,
                                    *gradient_matrix_y_);
  SparseMatrix::buildRectangularCsr(triplets_z, n_nodes, n_velocity_dof,
                                    *gradient_matrix_z_);
}

// P2 Stiffness Matrix: Integral of mu * (grad N_i : grad N_j) dVol
void NavierStokesSolver::buildStiffnessMatrix() {
  const size_t n_nodes = mesh_ptr_->nVertices();
  const size_t n_total_dof = mesh_ptr_->getP1plusP2DegreesOfFreedom();
  const size_t n_tets = mesh_ptr_->nTets();

  std::vector<std::tuple<int, int, double>> triplets;
  triplets.reserve(n_tets * 100);

  for (size_t e = 0; e < n_tets; ++e) {
    const Tet &tet = mesh_ptr_->tetAt(e);
    double vol = tet.volume;
    const auto &grad_lambda = mesh_ptr_->getTetGradient(e);

    // TODO: Map viscosity from nodes/elements properly. Using constant for now.
    double mu = fluid_properties_.viscosity;

    // Gather indices
    std::array<int, 10> ids;
    for (int i = 0; i < 4; ++i)
      ids[i] = tet.vertids[i];
    const auto &edge_nodes = mesh_ptr_->getTetEdgeNodes(e);
    for (int i = 0; i < 6; ++i)
      ids[4 + i] = n_nodes + edge_nodes[i];

    // --- Quadrature Loop ---
    // Accumulate into local matrix first to reduce triplet insertions
    std::array<std::array<double, 10>, 10> local_K = {{{0.0}}};

    for (const auto &qp : FEM_Utils::quadrature_rule) {
      auto dNdL = FEM_Utils::eval_grad_P2_wrt_lambda(qp.lambda);

      // Physical Gradients
      std::array<glm::dvec3, 10> grad_N;
      for (int i = 0; i < 10; ++i) {
        grad_N[i] = glm::dvec3(0.0);
        for (int k = 0; k < 4; ++k) {
          grad_N[i] += dNdL[i][k] * grad_lambda[k];
        }
      }

      double w = qp.w * vol * mu;

      for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
          local_K[i][j] += w * glm::dot(grad_N[i], grad_N[j]);
        }
      }
    }

    // Push to triplets
    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < 10; ++j) {
        if (std::abs(local_K[i][j]) > 1e-16) {
          triplets.emplace_back(ids[i], ids[j], local_K[i][j]);
        }
      }
    }
  }

  stiffness_matrix_ = std::make_unique<SparseMatrix>(n_total_dof);
  SparseMatrix::buildCsrFromTriplets(n_total_dof, triplets, *stiffness_matrix_,
                                     true);
}

// P2 Mass Matrix: Integral of rho * N_i * N_j dVol
void NavierStokesSolver::buildMassMatrix() {
  const size_t n_nodes = mesh_ptr_->nVertices();
  const size_t n_total_dof = mesh_ptr_->getP1plusP2DegreesOfFreedom();
  const size_t n_tets = mesh_ptr_->nTets();

  std::vector<std::tuple<int, int, double>> triplets;
  triplets.reserve(n_tets * 100);

  for (size_t e = 0; e < n_tets; ++e) {
    const Tet &tet = mesh_ptr_->tetAt(e);
    double vol = tet.volume;
    double rho = fluid_properties_.density;

    std::array<int, 10> ids;
    for (int i = 0; i < 4; ++i)
      ids[i] = tet.vertids[i];
    const auto &edge_nodes = mesh_ptr_->getTetEdgeNodes(e);
    for (int i = 0; i < 6; ++i)
      ids[4 + i] = n_nodes + edge_nodes[i];

    std::array<std::array<double, 10>, 10> local_M = {{{0.0}}};

    // --- Quadrature Loop ---
    for (const auto &qp : FEM_Utils::quadrature_rule) {
      auto N = FEM_Utils::eval_P2(qp.lambda);
      double w = qp.w * vol * rho;

      for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
          local_M[i][j] += w * N[i] * N[j];
        }
      }
    }

    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < 10; ++j) {
        if (std::abs(local_M[i][j]) > 1e-16) {
          triplets.emplace_back(ids[i], ids[j], local_M[i][j]);
        }
      }
    }
  }

  mass_matrix_ = std::make_unique<SparseMatrix>(n_total_dof);
  SparseMatrix::buildCsrFromTriplets(n_total_dof, triplets, *mass_matrix_,
                                     true);
}

// P1 Poisson Matrix (Pressure Correction): Integral of (grad phi_i : grad
// phi_j) dVol Since P1 gradients are constant, we don't strictly need a
// quadrature loop here, but the logic remains simple: Vol * dot(grad_i,
// grad_j).
void NavierStokesSolver::buildPoissonMatrix() {
  const size_t n_nodes = mesh_ptr_->nVertices();
  const size_t n_tets = mesh_ptr_->nTets();

  std::vector<std::tuple<int, int, double>> triplets;
  triplets.reserve(n_tets * 16);

  for (size_t e = 0; e < n_tets; ++e) {
    const Tet &tet = mesh_ptr_->tetAt(e);
    double vol = tet.volume;
    const auto &grad_N = mesh_ptr_->getTetGradient(e); // P1 Gradients

    for (int i = 0; i < 4; ++i) {
      int gi = tet.vertids[i];
      for (int j = 0; j < 4; ++j) {
        int gj = tet.vertids[j];

        // Exact integration for Linear elements
        double val = vol * glm::dot(grad_N[i], grad_N[j]);

        triplets.emplace_back(gi, gj, val);
      }
    }
  }

  poisson_matrix_ = std::make_unique<SparseMatrix>(n_nodes);
  SparseMatrix::buildCsrFromTriplets(n_nodes, triplets, *poisson_matrix_, true);

  // TODO: Enforce Dirichlet BCs
}
