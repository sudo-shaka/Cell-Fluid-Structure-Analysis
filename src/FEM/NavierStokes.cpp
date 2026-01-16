#include "FEM/NavierStokes.hpp"
#include "LinearAlgebra/LinearSolvers.hpp"
#include "LinearAlgebra/SparseMatrix.hpp"
#include "Mesh/Mesh.hpp"
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
  const size_t n_edges = mesh_ptr_->getNumberOfEdgeNodes();
  const size_t vert_dof = nv + n_edges;

  fluid_properties_ = fluid_props;
  fluid_properties_.effective_viscosity.resize(vert_dof, fluid_props.viscosity);

  velocity_.resize(vert_dof, glm::dvec3{0.0});
  velocity_star_.resize(vert_dof, glm::dvec3{0.0});
  pressure_.resize(nv, 0.0);
  pressure_correction_.resize(nv, 0.0);

  // 1. Build Matrices
  buildMassMatrix(); // Calculates lumped mass too
  buildStiffnessMatrix();
  buildGradientMatrices();
  buildPoissonMatrix();

  // 2. Apply Inlet Velocity to ALL Inlet Nodes (P1 Vertices + P2 Edges)
  // P1 Nodes
  for (size_t vi = 0; vi < nv; ++vi) {
    if (mesh_ptr_->getFluidVertexBC(vi) == FluidBCType::Inlet) {
      velocity_[vi] = mean_inlet_velocity_;
    }
  }
  // P2 Nodes
  for (size_t k = 0; k < n_edges; ++k) {
    if (mesh_ptr_->getP2FluidVertexBC(k) == FluidBCType::Inlet) {
      // P2 nodes are stored after P1 nodes
      velocity_[nv + k] = mean_inlet_velocity_;
    }
  }

  // 3. Set Reference Node
  if (reference_node_ < 0) {
    for (size_t vi = 0; vi < nv; ++vi) {
      if (mesh_ptr_->getFluidVertexBC(vi) == FluidBCType::Outlet) {
        reference_node_ = (int)vi;
        break;
      }
    }
  }
}

// HELPER: Compute Diagonal (Lumped) Mass Matrix
// We sum the rows of the consistent mass matrix to get a diagonal
// approximation. This is required for the explicit velocity correction step: u
// = u* - dt * inv_M * grad(p)
void NavierStokesSolver::computeLumpedMassInverse() {
  size_t n_dof = mesh_ptr_->getP1plusP2DegreesOfFreedom();
  inv_lumped_mass_.resize(n_dof, 0.0);

  // We can reconstruct this from the consistent mass_matrix_ we already built
  // or integrate it directly. Summing rows of the consistent matrix is
  // safest/easiest.

  // Iterate over all rows of the mass matrix
  for (size_t i = 0; i < n_dof; ++i) {
    double row_sum = 0.0;
    // Depending on your SparseMatrix implementation, iterate the row.
    // Assuming CSR format or similar standard access:
    for (const auto &entry : mass_matrix_->getRow(i)) {
      row_sum += entry.second;
    }

    // Safety check for zero rows (should not happen in proper FEM)
    if (std::abs(row_sum) < 1e-16) {
      inv_lumped_mass_[i] = 0.0;
    } else {
      inv_lumped_mass_[i] = 1.0 / row_sum;
    }
  }
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
// P2 Stiffness Matrix with Variable Viscosity
// Integral: mu(x) * (grad N_i : grad N_j) dVol
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

    // 1. Gather indices for 10 nodes (P2 element)
    std::array<int, 10> ids;
    for (int i = 0; i < 4; ++i)
      ids[i] = tet.vertids[i];
    const auto &edge_nodes = mesh_ptr_->getTetEdgeNodes(e);
    for (int i = 0; i < 6; ++i)
      ids[4 + i] = n_nodes + edge_nodes[i];

    // 2. Gather Nodal Viscosity values
    // Using P2 interpolation for viscosity allows it to vary quadratically
    // across the element
    std::array<double, 10> nodal_mu;
    for (int i = 0; i < 10; ++i) {
      nodal_mu[i] = fluid_properties_.effective_viscosity[ids[i]];
    }

    // Accumulate into local matrix first
    std::array<std::array<double, 10>, 10> local_K = {{{0.0}}};

    // --- Quadrature Loop ---
    for (const auto &qp : FEM_Utils::quadrature_rule) {

      // A. Interpolate Viscosity at the Quadrature Point
      // mu(x) = sum( N_k * mu_k )
      auto N = FEM_Utils::eval_P2(qp.lambda);
      double mu_qp = 0.0;
      for (int k = 0; k < 10; ++k) {
        mu_qp += N[k] * nodal_mu[k];
      }

      // B. Evaluate Gradients for Stiffness Terms
      auto dNdL = FEM_Utils::eval_grad_P2_wrt_lambda(qp.lambda);

      // Convert to Physical Gradients (dN/dx)
      std::array<glm::dvec3, 10> grad_N;
      for (int i = 0; i < 10; ++i) {
        grad_N[i] = glm::dvec3(0.0);
        for (int k = 0; k < 4; ++k) {
          grad_N[i] += dNdL[i][k] * grad_lambda[k];
        }
      }

      // Integration weight: w_q * Volume * Viscosity(x)
      double w = qp.w * vol * mu_qp;

      // C. Assembly: w * (grad_i . grad_j)
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

  // Resize lumped mass inverse vector
  inv_lumped_mass_.assign(n_total_dof, 0.0);

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

    // --- Quadrature Loop ---
    for (const auto &qp : FEM_Utils::quadrature_rule) {
      auto N = FEM_Utils::eval_P2(qp.lambda);
      double w = qp.w * vol * rho;

      for (int i = 0; i < 10; ++i) {
        // Build Lumped Mass (Diagonal scaling)
        // Accumulate row-sum for consistent mass approximation
        inv_lumped_mass_[ids[i]] += w * N[i];

        for (int j = 0; j < 10; ++j) {
          double val = w * N[i] * N[j];
          if (std::abs(val) > 1e-16) {
            triplets.emplace_back(ids[i], ids[j], val);
          }
        }
      }
    }
  }

  // Invert lumped mass for quick multiplication later
  for (double &val : inv_lumped_mass_) {
    if (val > 1e-20)
      val = 1.0 / val;
    else
      val = 0.0;
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
}

double NavierStokesSolver::getMeanFacePressure(size_t fi) const {
  assert(mesh_ptr_ && fi < mesh_ptr_->nFaces());
  const auto &faces = mesh_ptr_->getFaces();
  double average_pressure = 0.0;
  for (const auto &vid : faces[fi].vertids) {
    average_pressure += pressure_[vid];
  }
  return average_pressure / static_cast<double>(faces[fi].vertids.size());
}

glm::dvec3 NavierStokesSolver::getPressureForceAtFace(size_t fi) const {
  assert(mesh_ptr_ && fi < mesh_ptr_->nFaces());
  double pressure = getMeanFacePressure(fi);
  const Face &face = mesh_ptr_->getFaces()[fi];
  return -face.normal * face.area * pressure;
}

// Calculates the integral of: N_i * (u . grad) u
// Returns a vector of forces (acceleration * mass) for each node
std::vector<glm::dvec3> NavierStokesSolver::computeAdvectionRHS() {
  const size_t n_tets = mesh_ptr_->nTets();
  const size_t n_nodes = mesh_ptr_->nVertices();
  const size_t n_total_dof = mesh_ptr_->getP1plusP2DegreesOfFreedom();

  // Initialize output vector (Force per node)
  std::vector<glm::dvec3> advection_force(n_total_dof, glm::dvec3(0.0));

  // Reusable containers to avoid reallocation
  std::array<int, 10> v_idx;
  std::array<glm::dvec3, 10> nodal_u;
  std::array<glm::dvec3, 10> grad_N; // Physical gradients of shape functions

  for (size_t ti = 0; ti < n_tets; ++ti) {
    const auto &tet = mesh_ptr_->tetAt(ti);
    double vol = tet.volume;
    const auto &grad_lambda = mesh_ptr_->getTetGradient(ti);

    //  Gather Node Indices & Velocities for this Tet
    for (int i = 0; i < 4; ++i)
      v_idx[i] = tet.vertids[i];
    const auto &edge_nodes = mesh_ptr_->getTetEdgeNodes(ti);
    for (int i = 0; i < 6; ++i)
      v_idx[4 + i] = n_nodes + edge_nodes[i];

    for (int i = 0; i < 10; ++i) {
      nodal_u[i] = velocity_[v_idx[i]];
    }

    // Quadrature Loop
    for (const auto &qp : FEM_Utils::quadrature_rule) {

      // Evaluate Shape Functions (N) & Gradients (dN/dLambda)
      auto N = FEM_Utils::eval_P2(qp.lambda);
      auto dNdL = FEM_Utils::eval_grad_P2_wrt_lambda(qp.lambda);

      // Compute Physical Gradients of Shape Functions (dN/dX)
      for (int i = 0; i < 10; ++i) {
        grad_N[i] = glm::dvec3(0.0);
        for (int k = 0; k < 4; ++k) {
          grad_N[i] += dNdL[i][k] * grad_lambda[k];
        }
      }

      // Interpolate Velocity (u) and Velocity Gradient (grad_u) at QP
      glm::dvec3 u_qp(0.0);
      // du_dx, du_dy, du_dz (Rows of the gradient tensor)
      glm::dvec3 grad_u_x(0.0), grad_u_y(0.0), grad_u_z(0.0);

      for (int k = 0; k < 10; ++k) {
        u_qp += nodal_u[k] * N[k];

        // grad_u_component = sum( u_node_component * grad_N )
        grad_u_x += nodal_u[k].x * grad_N[k];
        grad_u_y += nodal_u[k].y * grad_N[k];
        grad_u_z += nodal_u[k].z * grad_N[k];
      }

      // Compute Convective Term: (u . grad) u
      // (u.grad)phi = u_x * dphi/dx + u_y * dphi/dy + u_z * dphi/dz
      double u_dot_grad_x = glm::dot(u_qp, grad_u_x);
      double u_dot_grad_y = glm::dot(u_qp, grad_u_y);
      double u_dot_grad_z = glm::dot(u_qp, grad_u_z);

      glm::dvec3 convective_term(u_dot_grad_x, u_dot_grad_y, u_dot_grad_z);

      // Assembly: Integral (N_i * convective_term)
      double weight =
          qp.w *
          vol; 

      for (int i = 0; i < 10; ++i) {
        // The convective term contributes as - (u . grad) u on the RHS.
        advection_force[v_idx[i]] += -weight * N[i] * convective_term;
      }
    }
  }

  return advection_force;
}

void NavierStokesSolver::computeDivergence(const std::vector<glm::dvec3> &u,
                                           std::vector<double> &div_out) {
  std::fill(div_out.begin(), div_out.end(), 0.0);
  std::vector<double> u_scalar(u.size());

  // X
  for (size_t i = 0; i < u.size(); ++i)
    u_scalar[i] = u[i].x;
  std::vector<double> res = gradient_matrix_x_->multiply(u_scalar);
  for (size_t i = 0; i < div_out.size(); ++i)
    div_out[i] += res[i];

  // Y
  for (size_t i = 0; i < u.size(); ++i)
    u_scalar[i] = u[i].y;
  res = gradient_matrix_y_->multiply(u_scalar);
  for (size_t i = 0; i < div_out.size(); ++i)
    div_out[i] += res[i];

  // Z
  for (size_t i = 0; i < u.size(); ++i)
    u_scalar[i] = u[i].z;
  res = gradient_matrix_z_->multiply(u_scalar);
  for (size_t i = 0; i < div_out.size(); ++i)
    div_out[i] += res[i];
}