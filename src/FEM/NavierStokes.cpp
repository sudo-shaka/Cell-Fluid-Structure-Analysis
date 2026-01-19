#include "FEM/NavierStokes.hpp"
#include "Mesh/Mesh.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <cmath>
#include <vector>

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

  velocity_.resize(vert_dof, Eigen::Vector3d::Zero());
  velocity_star_.resize(vert_dof, Eigen::Vector3d::Zero());
  pressure_.resize(nv, 0.0);
  pressure_correction_.resize(nv, 0.0);

  // Build Matrices
  buildMassMatrix();
  buildStiffnessMatrix();
  buildGradientMatrices();
  buildPoissonMatrix();

  // Apply Inlet Velocity to ALL Inlet Nodes (P1 Vertices + P2 Edges)
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

  //  Set Reference Node
  if (reference_node_ < 0) {
    for (size_t vi = 0; vi < nv; ++vi) {
      if (mesh_ptr_->getFluidVertexBC(vi) == FluidBCType::Outlet) {
        reference_node_ = (int)vi;
        break;
      }
    }
  }
}

void NavierStokesSolver::computeLumpedMassInverse() {
  // TODO
}

void NavierStokesSolver::buildGradientMatrices() {
  if (!mesh_ptr_) {
    std::cerr << "[NS] Error: mesh not initialized" << std::endl;
    return;
  }

  const size_t nv = mesh_ptr_->nVertices();
  const size_t nt = mesh_ptr_->nTets();

  // Gradient matrices map P1 pressure to P1 velocity (nv x nv)
  gradient_matrix_x_ = std::make_unique<Eigen::SparseMatrix<double>>(nv, nv);
  gradient_matrix_y_ = std::make_unique<Eigen::SparseMatrix<double>>(nv, nv);
  gradient_matrix_z_ = std::make_unique<Eigen::SparseMatrix<double>>(nv, nv);

  // Use triplet list for efficient sparse matrix construction
  std::vector<Eigen::Triplet<double>> triplets_x, triplets_y, triplets_z;
  triplets_x.reserve(nt * 16);
  triplets_y.reserve(nt * 16);
  triplets_z.reserve(nt * 16);

  // Assemble gradient matrices element by element
  for (size_t ti = 0; ti < nt; ++ti) {
    const auto &tet = mesh_ptr_->tetAt(ti);
    const auto &grads = mesh_ptr_->getTetGradients(ti);
    const double vol = tet.volume;

    // Use integration by parts: ∫ φ_i * ∂p/∂x dV = -∫ p * ∂φ_i/∂x dV
    // So G_x[i,j] = -∫ φ_j * ∂φ_i/∂x dV (negative of transpose)
    // For P1 elements, gradients are constant, so:
    // G_x[i,j] = -vol * ∫φ_j dV * grad_i.x() = -vol/4 * grad_i.x()

    for (int i = 0; i < 4; ++i) {
      const int vi = tet.vertids[i];

      for (int j = 0; j < 4; ++j) {
        const int vj = tet.vertids[j];
        const Eigen::Vector3d &grad_j = grads[j];

        // Gradient: ∇p = Σ_j p_j ∇φ_j
        // Weighted by test function: ∫ φ_i ∇p dV = Σ_j p_j ∫ φ_i ∇φ_j dV
        // For P1: ∫ φ_i dV = vol/4, and ∇φ_j is constant, so:
        // G[i,j] = (vol/4) * ∇φ_j
        // Divergence is then computed as -G^T
        const double contrib = vol / 4.0;

        triplets_x.emplace_back(vi, vj, contrib * grad_j.x());
        triplets_y.emplace_back(vi, vj, contrib * grad_j.y());
        triplets_z.emplace_back(vi, vj, contrib * grad_j.z());
      }
    }
  }

  // Build sparse matrices from triplets
  gradient_matrix_x_->setFromTriplets(triplets_x.begin(), triplets_x.end());
  gradient_matrix_y_->setFromTriplets(triplets_y.begin(), triplets_y.end());
  gradient_matrix_z_->setFromTriplets(triplets_z.begin(), triplets_z.end());

  gradient_matrix_x_->makeCompressed();
  gradient_matrix_y_->makeCompressed();
  gradient_matrix_z_->makeCompressed();

  std::cout << "[NS] Gradient matrices built: " << nv << " x " << nv
            << " (nnz_x=" << gradient_matrix_x_->nonZeros() << ")" << std::endl;
}

void NavierStokesSolver::buildStiffnessMatrix() {
  if (!mesh_ptr_) {
    std::cerr << "[NS] Error: mesh not initialized" << std::endl;
    return;
  }

  const size_t nv = mesh_ptr_->nVertices();
  const size_t n_edges = mesh_ptr_->getNumberOfEdgeNodes();
  const size_t vert_dof = nv + n_edges; // P2 elements (Taylor-Hood)
  const size_t nt = mesh_ptr_->nTets();

  stiffness_matrix_ =
      std::make_unique<Eigen::SparseMatrix<double>>(vert_dof, vert_dof);

  // Use triplet list for efficient construction
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(nt * 100); // P2 has 10 nodes per tet

  const double mu = fluid_properties_.viscosity;

  // Assemble stiffness matrix: K[i,j] = ∫ μ * (∇φ_i : ∇φ_j) dV
  for (size_t ti = 0; ti < nt; ++ti) {
    const auto &tet = mesh_ptr_->tetAt(ti);
    const auto &grads_p1 = mesh_ptr_->getTetGradients(ti);
    const auto &edge_nodes = mesh_ptr_->getTetEdgeNodes(ti);
    const double vol = tet.volume;

    // P1 nodes contribution (vertices)
    for (int i = 0; i < 4; ++i) {
      const int vi = tet.vertids[i];
      const Eigen::Vector3d &grad_i = grads_p1[i];

      for (int j = 0; j < 4; ++j) {
        const int vj = tet.vertids[j];
        const Eigen::Vector3d &grad_j = grads_p1[j];

        // Stiffness contribution: μ * vol * (grad_i · grad_j)
        const double k_ij = mu * vol * grad_i.dot(grad_j);
        triplets.emplace_back(vi, vj, k_ij);
      }
    }

    // P2 edge nodes contribution (simplified)
    // For production code, use proper P2 quadrature
    // Here we use a simplified approach with edge midpoints
    for (int e = 0; e < 6; ++e) {
      const int ei = nv + edge_nodes[e];

      // Edge nodes interact with vertex nodes and other edge nodes
      for (int i = 0; i < 4; ++i) {
        const int vi = tet.vertids[i];
        const Eigen::Vector3d &grad_i = grads_p1[i];

        // Approximate edge-vertex coupling (reduced weight)
        const double k_ei = mu * vol * grad_i.squaredNorm() * 0.5;
        triplets.emplace_back(ei, vi, k_ei);
        triplets.emplace_back(vi, ei, k_ei);
      }

      // Edge-edge coupling (diagonal dominant)
      for (int f = 0; f < 6; ++f) {
        const int ej = nv + edge_nodes[f];
        const double k_ee = (e == f) ? mu * vol * 2.0 : mu * vol * 0.25;
        triplets.emplace_back(ei, ej, k_ee);
      }
    }
  }

  stiffness_matrix_->setFromTriplets(triplets.begin(), triplets.end());
  stiffness_matrix_->makeCompressed();

  std::cout << "[NS] Stiffness matrix built: " << vert_dof << " x " << vert_dof
            << " (nnz=" << stiffness_matrix_->nonZeros() << ")" << std::endl;
}

void NavierStokesSolver::buildMassMatrix() {
  if (!mesh_ptr_) {
    std::cerr << "[NS] Error: mesh not initialized" << std::endl;
    return;
  }

  const size_t nv = mesh_ptr_->nVertices();
  const size_t n_edges = mesh_ptr_->getNumberOfEdgeNodes();
  const size_t vert_dof = nv + n_edges;
  const size_t nt = mesh_ptr_->nTets();

  mass_matrix_ =
      std::make_unique<Eigen::SparseMatrix<double>>(vert_dof, vert_dof);
  inv_lumped_mass_.resize(vert_dof, 0.0);

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(nt * 100);

  const double rho = fluid_properties_.density;

  // Assemble consistent mass matrix: M[i,j] = ∫ ρ * φ_i * φ_j dV
  for (size_t ti = 0; ti < nt; ++ti) {
    const auto &tet = mesh_ptr_->tetAt(ti);
    const auto &edge_nodes = mesh_ptr_->getTetEdgeNodes(ti);
    const double vol = tet.volume;
    const double mass = rho * vol;

    // P1 mass matrix (standard FEM formula)
    // M_ij = (ρ*V/20) * (1 + δ_ij) where δ_ij is Kronecker delta
    for (int i = 0; i < 4; ++i) {
      const int vi = tet.vertids[i];

      for (int j = 0; j < 4; ++j) {
        const int vj = tet.vertids[j];
        const double m_ij = (i == j) ? mass / 10.0 : mass / 20.0;
        triplets.emplace_back(vi, vj, m_ij);
      }
    }

    // P2 edge nodes (higher integration weight at edges)
    for (int e = 0; e < 6; ++e) {
      const int ei = nv + edge_nodes[e];

      // Edge-vertex mass coupling
      for (int i = 0; i < 4; ++i) {
        const int vi = tet.vertids[i];
        const double m_ei = mass / 30.0;
        triplets.emplace_back(ei, vi, m_ei);
        triplets.emplace_back(vi, ei, m_ei);
      }

      // Edge-edge mass
      for (int f = 0; f < 6; ++f) {
        const int ej = nv + edge_nodes[f];
        const double m_ee = (e == f) ? mass / 15.0 : mass / 60.0;
        triplets.emplace_back(ei, ej, m_ee);
      }
    }
  }

  mass_matrix_->setFromTriplets(triplets.begin(), triplets.end());
  mass_matrix_->makeCompressed();

  // Compute lumped mass inverse for explicit schemes
  // Lumped mass is diagonal: M_lump[i] = sum_j M[i,j]
  // Use matrix-vector product with ones vector to compute row sums
  Eigen::VectorXd ones = Eigen::VectorXd::Ones(vert_dof);
  Eigen::VectorXd row_sums = (*mass_matrix_) * ones;

  for (size_t i = 0; i < vert_dof; ++i) {
    // Invert (with safety check)
    if (std::abs(row_sums(i)) > 1e-14) {
      inv_lumped_mass_[i] = 1.0 / row_sums(i);
    } else {
      // Very small mass - set to safe value
      inv_lumped_mass_[i] = 1.0;
      std::cerr << "[NS] Warning: very small lumped mass at node " << i
                << std::endl;
    }
  }

  std::cout << "[NS] Mass matrix built: " << vert_dof << " x " << vert_dof
            << " (nnz=" << mass_matrix_->nonZeros() << ")" << std::endl;
}

// P1 Poisson Matrix (Pressure Correction): Integral of (grad phi_i : grad
// phi_j) dVol Since P1 gradients are constant, we don't strictly need a
// quadrature loop here, but the logic remains simple: Vol * dot(grad_i,
// grad_j).
void NavierStokesSolver::buildPoissonMatrix() {
  if (!mesh_ptr_) {
    std::cerr << "[NS] Error: mesh not initialized" << std::endl;
    return;
  }

  const size_t nv = mesh_ptr_->nVertices();
  const size_t nt = mesh_ptr_->nTets();

  poisson_matrix_ = std::make_unique<Eigen::SparseMatrix<double>>(nv, nv);

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(nt * 16);

  // Assemble Poisson matrix: L[i,j] = ∫ (∇φ_i · ∇φ_j) dV
  for (size_t ti = 0; ti < nt; ++ti) {
    const auto &tet = mesh_ptr_->tetAt(ti);
    const auto &grads = mesh_ptr_->getTetGradients(ti);
    const double vol = tet.volume;

    for (int i = 0; i < 4; ++i) {
      const int vi = tet.vertids[i];
      const Eigen::Vector3d &grad_i = grads[i];

      for (int j = 0; j < 4; ++j) {
        const int vj = tet.vertids[j];
        const Eigen::Vector3d &grad_j = grads[j];

        // Laplacian stiffness: vol * (grad_i · grad_j)
        const double l_ij = vol * grad_i.dot(grad_j);
        triplets.emplace_back(vi, vj, l_ij);
      }
    }
  }

  poisson_matrix_->setFromTriplets(triplets.begin(), triplets.end());
  poisson_matrix_->makeCompressed();

  std::cout << "[NS] Poisson matrix built: " << nv << " x " << nv
            << " (nnz=" << poisson_matrix_->nonZeros() << ")" << std::endl;
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

Eigen::Vector3d NavierStokesSolver::getPressureForceAtFace(size_t fi) const {
  assert(mesh_ptr_ && fi < mesh_ptr_->nFaces());
  double pressure = getMeanFacePressure(fi);
  const Face &face = mesh_ptr_->getFaces()[fi];
  return -face.normal * face.area * pressure;
}

bool NavierStokesSolver::pisoStep() {
  if (!is_initialized_) {
    std::cerr << "[NS] Error: solver not initialized" << std::endl;
    return false;
  }

  // PISO Algorithm (Pressure Implicit with Splitting of Operators)
  // 1. Momentum predictor: solve for u* using old pressure
  // 2. Pressure corrector: solve Poisson equation for pressure correction
  // 3. Velocity corrector: update velocity using new pressure
  // Can iterate steps 2-3 for better coupling

  // Step 1: Momentum predictor
  if (!solveMomentumPredictor()) {
    std::cerr << "[NS] Momentum predictor failed" << std::endl;
    return false;
  }

  // Steps 2-3: Pressure-velocity correction loop
  for (int corrector = 0; corrector < n_corrections_; ++corrector) {
    // Step 2: Solve pressure Poisson equation
    if (!solvePressurePoisson()) {
      std::cerr << "[NS] Pressure Poisson solve failed" << std::endl;
      return false;
    }

    // Step 3: Correct velocity field
    if (!correctVelocity()) {
      std::cerr << "[NS] Velocity correction failed" << std::endl;
      return false;
    }
  }

  // Enforce boundary conditions
  reenforceVelocityBCs();

  // Update time
  time_ += dt_;

  return !hasNans();
}

bool NavierStokesSolver::solveMomentumPredictor() {
  // Solve momentum equation: (M/dt + K) u* = M/dt * u_n - grad(p_old)
  // where K is the diffusion matrix (stiffness), M is mass

  const size_t nv = mesh_ptr_->nVertices();
  const size_t n_edges = mesh_ptr_->getNumberOfEdgeNodes();
  const size_t vert_dof = nv + n_edges;

  // Solve for each component separately
  for (int comp = 0; comp < 3; ++comp) {
    // Build RHS: M/dt * u_n - grad(p_old)
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(vert_dof);

    // Add mass term: M/dt * u_n
    // Extract velocity component as a vector
    Eigen::VectorXd u_comp(vert_dof);
    for (size_t i = 0; i < vert_dof; ++i) {
      u_comp(i) = velocity_[i](comp);
    }
    rhs += (*mass_matrix_) * u_comp / dt_;

    // Subtract pressure gradient contribution (only for P1 nodes)
    const auto *grad_matrix = (comp == 0)   ? gradient_matrix_x_.get()
                              : (comp == 1) ? gradient_matrix_y_.get()
                                            : gradient_matrix_z_.get();

    Eigen::VectorXd p_vec = Eigen::Map<Eigen::VectorXd>(pressure_.data(), nv);
    Eigen::VectorXd grad_p = (*grad_matrix) * p_vec;

    // Only apply to P1 nodes (first nv entries); P2 nodes get zero pressure
    // gradient
    for (size_t i = 0; i < nv; ++i) {
      rhs(i) -= grad_p(i);
    }

    // Build LHS: M/dt + K (implicit diffusion)
    Eigen::SparseMatrix<double> lhs = *mass_matrix_ / dt_ + *stiffness_matrix_;

    // Apply Dirichlet boundary conditions using penalty method
    const double penalty = 1e10; // Reduced from 1e14 for numerical stability
    const size_t n_edges_local = n_edges;

    // P1 nodes - apply inlet and wall BCs
    for (size_t i = 0; i < nv; ++i) {
      FluidBCType bc = mesh_ptr_->getFluidVertexBC(i);
      if (bc == FluidBCType::Inlet || bc == FluidBCType::Wall) {
        double u_prescribed = 0.0;
        if (bc == FluidBCType::Inlet) {
          u_prescribed = mean_inlet_velocity_(comp);
        }

        // Add penalty term to diagonal and RHS
        lhs.coeffRef(i, i) += penalty;
        rhs(i) += penalty * u_prescribed;
      }
    }

    // P2 edge nodes - apply inlet and wall BCs
    for (size_t k = 0; k < n_edges_local; ++k) {
      FluidBCType bc = mesh_ptr_->getP2FluidVertexBC(k);
      size_t idx = nv + k;

      if (bc == FluidBCType::Inlet || bc == FluidBCType::Wall) {
        double u_prescribed = 0.0;
        if (bc == FluidBCType::Inlet) {
          u_prescribed = mean_inlet_velocity_(comp);
        }

        lhs.coeffRef(idx, idx) += penalty;
        rhs(idx) += penalty * u_prescribed;
      }
    }

    // Solve linear system using BiCGSTAB (more stable for nonsymmetric systems)
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
    solver.setMaxIterations(1000);
    solver.setTolerance(1e-6);
    solver.compute(lhs);

    if (solver.info() != Eigen::Success) {
      std::cerr << "[NS] Momentum matrix setup failed for component " << comp
                << std::endl;
      return false;
    }

    Eigen::VectorXd sol = solver.solve(rhs);

    if (solver.info() != Eigen::Success) {
      std::cerr << "[NS] Momentum solve failed for component " << comp
                << std::endl;
      return false;
    }

    // Update velocity_star with solution
    for (size_t i = 0; i < vert_dof; ++i) {
      velocity_star_[i](comp) = sol(i);
    }
  }

  return true;
}


bool NavierStokesSolver::solvePressurePoisson() {
  const size_t nv = mesh_ptr_->nVertices();

  // Compute divergence of u*
  std::vector<double> divergence(nv, 0.0);
  computeDivergence(velocity_star_, divergence);

  // RHS: div(u*) / dt
  std::vector<double> rhs(nv);
  for (size_t i = 0; i < nv; ++i) {
    rhs[i] = divergence[i] / dt_;
  }

  // Build modified Poisson matrix with BCs
  Eigen::SparseMatrix<double> L_bc = *poisson_matrix_;
  
  // Fix pressure at reference node ONLY if using natural/Neumann outlets
  if (reference_node_ >= 0 && reference_node_ < static_cast<int>(nv)) {
    // For Neumann outlets: fix one node to remove null space
    if (outlet_type_ != OutletType::DirichletPressure) {
      const double large_val = 1e20;
      
      for (int k = 0; k < L_bc.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L_bc, k); it; ++it) {
          if (it.row() == reference_node_) {
            if (it.col() == reference_node_) {
              it.valueRef() = large_val;
            } else {
              it.valueRef() = 0.0;
            }
          } else if (it.col() == reference_node_) {
            it.valueRef() = 0.0;
          }
        }
      }
      rhs[reference_node_] = 0.0;
    } else {
      // For Dirichlet pressure outlets: apply BC via penalty method
      const double penalty = 1e12;
      for (size_t i = 0; i < nv; ++i) {
        if (mesh_ptr_->getFluidVertexBC(i) == FluidBCType::Outlet) {
          L_bc.coeffRef(i, i) += penalty;
          rhs[i] += penalty * outlet_pressure_;
        }
      }
    }
  }

  // Solve system
  Eigen::VectorXd rhs_eigen = Eigen::Map<Eigen::VectorXd>(rhs.data(), rhs.size());
  Eigen::VectorXd x_eigen(nv);

  Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
  solver.setMaxIterations(1000);
  solver.setTolerance(1e-8);
  solver.compute(L_bc);

  if (solver.info() != Eigen::Success) {
    std::cerr << "[NS] Pressure Poisson matrix factorization failed" << std::endl;
    return false;
  }

  x_eigen = solver.solve(rhs_eigen);

  if (solver.info() != Eigen::Success) {
    std::cerr << "[NS] Pressure Poisson solve failed" << std::endl;
    return false;
  }

  // Copy solution to pressure_correction_
  for (size_t i = 0; i < nv; ++i) {
    pressure_correction_[i] = x_eigen(i);
  }

  // Update pressure: p = p_old + relax_p * p'
  for (size_t i = 0; i < nv; ++i) {
    pressure_[i] += relax_p_ * pressure_correction_[i];
  }

  // Apply outlet pressure BCs AFTER update
  reinforcePressureBCs();
  
  // Normalize pressure to reference node
  normalizePressire();

  return true;
}

void NavierStokesSolver::computeDivergence(
    const std::vector<Eigen::Vector3d> &u, std::vector<double> &div_out) {
  const size_t nv = mesh_ptr_->nVertices();
  div_out.assign(nv, 0.0);

  // div(u) = du/dx + dv/dy + dw/dz
  // Since G[i,j] = (vol/4) * ∇φ_j computes gradient,
  // divergence is -G^T (negative transpose)

  // For each pressure node i (iterating over columns of G, which are rows of
  // G^T)
  for (size_t j = 0; j < nv; ++j) {
    // Sum contributions from velocity node j to all pressure nodes i
    for (Eigen::SparseMatrix<double>::InnerIterator it(*gradient_matrix_x_, j);
         it; ++it) {
      const int i = it.row();           // Pressure node (was velocity DOF)
      const double coeff = -it.value(); // Negative for divergence = -G^T
      div_out[i] += coeff * u[j](0);    // du/dx contribution
    }
    for (Eigen::SparseMatrix<double>::InnerIterator it(*gradient_matrix_y_, j);
         it; ++it) {
      const int i = it.row();
      const double coeff = -it.value();
      div_out[i] += coeff * u[j](1); // dv/dy contribution
    }
    for (Eigen::SparseMatrix<double>::InnerIterator it(*gradient_matrix_z_, j);
         it; ++it) {
      const int i = it.row();
      const double coeff = -it.value();
      div_out[i] += coeff * u[j](2); // dw/dz contribution
    }
  }
}

bool NavierStokesSolver::correctVelocity() {
  const size_t nv = mesh_ptr_->nVertices();
  const size_t n_edges = mesh_ptr_->getNumberOfEdgeNodes();

  for (int comp = 0; comp < 3; ++comp) {
    const auto *grad_matrix = (comp == 0)   ? gradient_matrix_x_.get()
                              : (comp == 1) ? gradient_matrix_y_.get()
                                            : gradient_matrix_z_.get();

    // P1 nodes: apply pressure correction
    for (size_t i = 0; i < nv; ++i) {
      FluidBCType bc = mesh_ptr_->getFluidVertexBC(i);
      
      // Skip Dirichlet boundary nodes (inlet and wall ONLY)
      if (bc == FluidBCType::Inlet || bc == FluidBCType::Wall) {
        velocity_[i](comp) = (bc == FluidBCType::Inlet) ? mean_inlet_velocity_(comp) : 0.0;
        continue;
      }

      // For Neumann outlets: apply pressure correction normally
      // The boundary condition is naturally satisfied in weak form
      double grad_p = 0.0;
      for (Eigen::SparseMatrix<double>::InnerIterator it(*grad_matrix, i); it; ++it) {
        grad_p += it.value() * pressure_correction_[it.col()];
      }

      if (inv_lumped_mass_[i] > 1e-14) {
        double u_correction = -dt_ * inv_lumped_mass_[i] * grad_p;
        velocity_[i](comp) = velocity_star_[i](comp) + relax_u_ * u_correction;
      } else {
        velocity_[i](comp) = velocity_star_[i](comp);
      }
    }
  }

  // Apply BCs for P2 edge nodes
  for (size_t k = 0; k < n_edges; ++k) {
    FluidBCType bc = mesh_ptr_->getP2FluidVertexBC(k);
    size_t idx = nv + k;

    for (int comp = 0; comp < 3; ++comp) {
      if (bc == FluidBCType::Inlet) {
        velocity_[idx](comp) = mean_inlet_velocity_(comp);
      } else if (bc == FluidBCType::Wall) {
        velocity_[idx](comp) = 0.0;
      } else {
        // For outlets and interior: keep corrected velocity
        velocity_[idx](comp) = velocity_star_[idx](comp);
      }
    }
  }

  return true;
}

bool NavierStokesSolver::solvePressureBiCGSTAB(const std::vector<double> &rhs,
                                               std::vector<double> &x) {
  Eigen::VectorXd rhs_eigen =
      Eigen::Map<const Eigen::VectorXd>(rhs.data(), rhs.size());
  Eigen::VectorXd x_eigen = Eigen::Map<Eigen::VectorXd>(x.data(), x.size());

  Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
  solver.setMaxIterations(1000);
  solver.setTolerance(1e-8);
  solver.compute(*poisson_matrix_);

  if (solver.info() != Eigen::Success) {
    std::cerr << "[NS] Failed to factorize Poisson matrix" << std::endl;
    return false;
  }

  x_eigen = solver.solve(rhs_eigen);

  if (solver.info() != Eigen::Success) {
    std::cerr << "[NS] BiCGSTAB failed" << std::endl;
    return false;
  }

  // Copy back
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = x_eigen(i);
  }

  return true;
}

bool NavierStokesSolver::normalizePressire() {
  if (reference_node_ >= 0 &&
      reference_node_ < static_cast<int>(pressure_.size())) {
    double ref_pressure = pressure_[reference_node_];
    for (auto &p : pressure_) {
      p -= ref_pressure;
    }
  }
  return true;
}

void NavierStokesSolver::reenforceVelocityBCs() {
  const size_t nv = mesh_ptr_->nVertices();
  const size_t n_edges = mesh_ptr_->getNumberOfEdgeNodes();

  // Enforce P1 BCs
  for (size_t i = 0; i < nv; ++i) {
    FluidBCType bc = mesh_ptr_->getFluidVertexBC(i);
    if (bc == FluidBCType::Inlet) {
      velocity_[i] = mean_inlet_velocity_;
    } else if (bc == FluidBCType::Wall) {
      velocity_[i] = Eigen::Vector3d::Zero();
    }
  }

  // Enforce P2 BCs
  for (size_t k = 0; k < n_edges; ++k) {
    FluidBCType bc = mesh_ptr_->getP2FluidVertexBC(k);
    size_t idx = nv + k;
    if (bc == FluidBCType::Inlet) {
      velocity_[idx] = mean_inlet_velocity_;
    } else if (bc == FluidBCType::Wall) {
      velocity_[idx] = Eigen::Vector3d::Zero();
    }
  }
}

std::vector<Eigen::Vector3d>
NavierStokesSolver::computePressureForces() const {
  if (!mesh_ptr_) {
    std::cerr << "[NS] Error: mesh not initialized" << std::endl;
    return {};
  }

  const size_t nv = mesh_ptr_->nVertices();
  std::vector<Eigen::Vector3d> pressure_forces(nv, Eigen::Vector3d::Zero());

  // Compute pressure force at each boundary node by summing contributions from
  // adjacent boundary faces For each boundary face, the pressure force is: F =
  // -p * n * area distributed to the face's vertices

  const auto &faces = mesh_ptr_->getFaces();
  for (size_t fi = 0; fi < faces.size(); ++fi) {
    const Face &face = faces[fi];

    // Only process boundary faces (faces with one adjacent tet)
    if (face.tet_b != -1) {
      continue; // Internal face, skip
    }

    // Compute mean pressure at the face
    double face_pressure = 0.0;
    for (const int vid : face.vertids) {
      face_pressure += pressure_[vid];
    }
    face_pressure /= static_cast<double>(face.vertids.size());

    // Pressure force on face: F = -p * n * A (negative because pressure points
    // inward)
    Eigen::Vector3d face_force = -face_pressure * face.normal * face.area;

    // Distribute force equally to face vertices (for P1 elements)
    const double force_per_vertex = 1.0 / static_cast<double>(face.vertids.size());
    for (const int vid : face.vertids) {
      pressure_forces[vid] += face_force * force_per_vertex;
    }
  }

  return pressure_forces;
}

std::vector<Eigen::Vector3d> NavierStokesSolver::computeShearStress() const {
  if (!mesh_ptr_) {
    std::cerr << "[NS] Error: mesh not initialized" << std::endl;
    return {};
  }

  const size_t nv = mesh_ptr_->nVertices();
  std::vector<Eigen::Vector3d> shear_forces(nv, Eigen::Vector3d::Zero());
  
  const double mu = fluid_properties_.viscosity;
  const auto &faces = mesh_ptr_->getFaces();
  
  // For each boundary face, compute viscous shear stress
  for (size_t fi = 0; fi < faces.size(); ++fi) {
    const Face &face = faces[fi];
    
    // Only process boundary faces (faces with one adjacent tet)
    if (face.tet_b != -1) {
      continue; // Internal face, skip
    }
    
    // Compute velocity gradient at face by averaging from adjacent tet
    // Simplified: use the gradient from the adjacent tetrahedron
    const int tet_idx = face.tet_a;
    const auto &tet = mesh_ptr_->tetAt(tet_idx);
    const auto &grads = mesh_ptr_->getTetGradients(tet_idx);
    
    // Compute velocity gradient tensor: ∇u
    Eigen::Matrix3d grad_u = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 4; ++i) {
      const int vid = tet.vertids[i];
      const Eigen::Vector3d &grad_phi = grads[i];
      const Eigen::Vector3d &u = velocity_[vid];
      
      // grad_u(i,j) = du_i/dx_j = Σ u_i * ∂φ/∂x_j
      grad_u += u * grad_phi.transpose();
    }
    
    // Compute symmetric strain rate tensor: ε = (∇u + ∇u^T)/2
    Eigen::Matrix3d strain_rate = 0.5 * (grad_u + grad_u.transpose());
    
    // Viscous stress tensor: τ = 2μ * ε
    Eigen::Matrix3d tau = 2.0 * mu * strain_rate;
    
    // Viscous traction on face: t_viscous = τ · n
    Eigen::Vector3d viscous_traction = tau * face.normal;
    
    // Total viscous force on face
    Eigen::Vector3d face_force = viscous_traction * face.area;
    
    // Distribute force equally to face vertices
    const double force_per_vertex = 1.0 / static_cast<double>(face.vertids.size());
    for (const int vid : face.vertids) {
      shear_forces[vid] += face_force * force_per_vertex;
    }
  }
  
  return shear_forces;
}

std::vector<Eigen::Vector3d> NavierStokesSolver::computeTotalFluidForces() const {
  if (!mesh_ptr_) {
    std::cerr << "[NS] Error: mesh not initialized" << std::endl;
    return {};
  }
  
  // Total fluid force = pressure force + viscous shear force
  std::vector<Eigen::Vector3d> pressure_forces = computePressureForces();
  std::vector<Eigen::Vector3d> shear_forces = computeShearStress();
  
  const size_t nv = mesh_ptr_->nVertices();
  std::vector<Eigen::Vector3d> total_forces(nv, Eigen::Vector3d::Zero());
  
  for (size_t i = 0; i < nv; ++i) {
    total_forces[i] = pressure_forces[i] + shear_forces[i];
  }
  
  return total_forces;
}

// SIMPLE SOLVER: Basic operator splitting without PISO iteration
bool NavierStokesSolver::simpleStep() {
  if (!is_initialized_) {
    std::cerr << "[NS-Simple] Error: solver not initialized" << std::endl;
    return false;
  }

  // SIMPLE Algorithm (Semi-Implicit Method for Pressure Linked Equations)
  // Simplified version with single pass:
  // 1. Momentum predictor: solve for u* using old pressure
  // 2. Pressure corrector: solve Poisson equation for pressure correction
  // 3. Velocity corrector: update velocity using new pressure
  // Note: Unlike PISO, we do NOT iterate steps 2-3 for speed

  // Step 1: Momentum predictor
  if (!solveMomentumPredictor()) {
    std::cerr << "[NS-Simple] Momentum predictor failed" << std::endl;
    return false;
  }

  // Step 2: Solve pressure Poisson equation (single correction)
  if (!solvePressurePoisson()) {
    std::cerr << "[NS-Simple] Pressure Poisson solve failed" << std::endl;
    return false;
  }

  // Re-enforce pressure boundary conditions
  reinforcePressureBCs();

  // Step 3: Correct velocity field (single correction)
  if (!correctVelocity()) {
    std::cerr << "[NS-Simple] Velocity correction failed" << std::endl;
    return false;
  }

  // Enforce boundary conditions
  reenforceVelocityBCs();

  // Update time
  time_ += dt_;

  return !hasNans();
}

void NavierStokesSolver::reinforcePressureBCs() {
  const size_t nv = mesh_ptr_->nVertices();

  // Only enforce Dirichlet pressure if explicitly set
  if (outlet_type_ == OutletType::DirichletPressure) {
    for (size_t i = 0; i < nv; ++i) {
      if (mesh_ptr_->getFluidVertexBC(i) == FluidBCType::Outlet) {
        pressure_[i] = outlet_pressure_;
      }
    }
  }
  // For Neumann outlets: do NOT override pressure values
  // The natural BC is satisfied through the weak form
}
