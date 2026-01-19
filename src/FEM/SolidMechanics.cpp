#include "FEM/SolidMechanics.hpp"
#include "Mesh/Mesh.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cassert>
#include <memory>

Material Material::linear_elastic(double youngs_modulus, double poissons_ratio,
                                  double density) {
  Material mat;
  mat.model = MaterialModel::LinearElastic;
  mat.youngs_modulus = youngs_modulus;
  mat.poissons_ratio = poissons_ratio;
  mat.density = density;

  mat.mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio));
  mat.lambda = youngs_modulus * poissons_ratio /
               ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio));
  mat.shear_modulus = mat.mu;
  mat.bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poissons_ratio));

  mat.c10 = 0.0;
  mat.c01 = 0.0;
  mat.alpha_m = 0.0;
  mat.alpha_k = 0.0;

  return mat;
}

Material Material::neo_hookean(double youngs_modulus, double poissons_ratio,
                               double density) {
  Material mat =
      Material::linear_elastic(youngs_modulus, poissons_ratio, density);
  mat.model = MaterialModel::NeoHookean;
  return mat;
}

Material Material::with_damping(double damping_m, double damping_k) const {
  Material mat = *this;
  mat.alpha_m = damping_m;
  mat.alpha_k = damping_k;
  return mat;
}

std::array<std::array<double, 6>, 6> Material::elasticity_tensor() const {
  double e = youngs_modulus;
  double nu = poissons_ratio;
  double factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu));

  std::array<std::array<double, 6>, 6> c = {};

  // Normal components
  c[0][0] = factor * (1.0 - nu);
  c[1][1] = factor * (1.0 - nu);
  c[2][2] = factor * (1.0 - nu);

  // Off-diagonal normal
  c[0][1] = factor * nu;
  c[0][2] = factor * nu;
  c[1][0] = factor * nu;
  c[1][2] = factor * nu;
  c[2][0] = factor * nu;
  c[2][1] = factor * nu;

  // Shear components
  c[3][3] = factor * (1.0 - 2.0 * nu) / 2.0;
  c[4][4] = factor * (1.0 - 2.0 * nu) / 2.0;
  c[5][5] = factor * (1.0 - 2.0 * nu) / 2.0;

  return c;
}

// SolidMechanicsSolver implementation
SolidMechanicsSolver::SolidMechanicsSolver()
    : is_initialized_(false),
      material_(Material::linear_elastic(1e6, 0.3, 1000.0)), mesh_(nullptr),
      max_iter_(1000), tolerance_(1e-8), dt_(1e-3), time_(0.0), beta_(0.25),
      gamma_(0.5), newton_max_iter_(20), newton_tolerance_(1e-6),
      displacement_relaxation_(0.2), displacement_smoothing_iters_(2),
      displacement_smoothing_factor_(0.3), max_displacement_factor_(0.1) {}

void SolidMechanicsSolver::initialize(std::shared_ptr<Mesh> mesh_ptr,
                                      const Material &mat) {
  std::cout << "[Solid] Initializing solver." << std::endl;
  mesh_ = mesh_ptr;
  material_ = mat;
  original_positions_ = mesh_->getVertPositions();

  size_t nv = mesh_->nVertices();
  displacement_.resize(nv, Eigen::Vector3d::Zero());
  velocity_.resize(nv, Eigen::Vector3d::Zero());
  acceleration_.resize(nv, Eigen::Vector3d::Zero());
  displacement_prev_.resize(nv, Eigen::Vector3d::Zero());
  velocity_prev.resize(nv, Eigen::Vector3d::Zero());
  acceleration_prev_.resize(nv, Eigen::Vector3d::Zero());
  total_displacement_.resize(nv, Eigen::Vector3d::Zero());
  displaced_positions_.resize(nv);
  for (size_t i = 0; i < nv; ++i) {
    displaced_positions_[i] = original_positions_[i];
  }
  strain_tensor_.resize(nv);
  stress_tensor_.resize(nv);
  von_mises_stress_.resize(nv, 0.0);
  principle_stresses_.resize(nv, Eigen::Vector3d::Zero());
  body_force_.resize(nv, Eigen::Vector3d::Zero());
  surface_traction_.resize(nv, Eigen::Vector3d::Zero());
  assembleMassMatrix();
  assembleStiffnessMatrix();
  is_initialized_ = true;
  std::cout << "[Solid] Initialization complete." << std::endl;
}

void SolidMechanicsSolver::applyGravity(Eigen::Vector3d g) {
  if (!mesh_)
    return;

  // Reset body force before applying gravity
  std::fill(body_force_.begin(), body_force_.end(), Eigen::Vector3d::Zero());

  for (const auto &tet : mesh_->getTets()) {
    double vol = tet.volume;
    double nodal_mass = material_.density * vol / 4.0; // Lumped mass per node

    for (int i = 0; i < 4; ++i) {
      int v_id = tet.vertids[i];
      body_force_[v_id] += nodal_mass * g;
    }
  }
}

void SolidMechanicsSolver::applyPressure(double pressure) {
  if (!mesh_)
    return;
  size_t nv = mesh_->nVertices();
  std::vector<double> pressures(nv, pressure);
  applyPressure(pressures);
}
void SolidMechanicsSolver::applyPressure(const std::vector<double> &pressures) {
  assert(mesh_);
  assert(pressures.size() == surface_traction_.size());
  size_t nv = mesh_->nVertices();
  std::vector<Eigen::Vector3d> global_forces(nv, Eigen::Vector3d::Zero());
  for (const auto &f : mesh_->getFaces()) {
    if (!f.is_ecm)
      continue;
    for (const auto &i : f.vertids) {
      if (mesh_->getSolidVertexBC(i) == SolidBCType::Fixed)
        continue;
      Eigen::Vector3d traction = pressures[i] * f.normal;
      global_forces[i] +=
          traction * f.area / static_cast<double>(f.vertids.size());
    }
  }
  for (size_t vi = 0; vi < mesh_->nVertices(); vi++)
    surface_traction_[vi] += global_forces[vi];
}

void SolidMechanicsSolver::addBodyForce(size_t node_id,
                                        const Eigen::Vector3d &force) {
  assert(node_id < body_force_.size());
  body_force_[node_id] += force;
}

void SolidMechanicsSolver::assembleMassMatrix() {
  if (!mesh_) {
    std::cerr << "[Solid] Error: mesh not initialized" << std::endl;
    return;
  }

  const size_t nv = mesh_->nVertices();
  const size_t nt = mesh_->nTets();

  // 3 DOF per node (x, y, z displacements)
  const size_t total_dof = 3 * nv;
  mass_matrix_.resize(total_dof, total_dof);

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(nt * 144); // 4 nodes * 3 DOF * 4 nodes * 3 DOF per tet

  const double rho = material_.density;

  // Assemble mass matrix element by element
  for (size_t ti = 0; ti < nt; ++ti) {
    const auto &tet = mesh_->tetAt(ti);
    const double vol = tet.volume;
    const double mass = rho * vol;

    // Consistent mass matrix for tetrahedral element
    // M_ij = (ρ*V/20) * (1 + δ_ij) for each DOF
    for (int i = 0; i < 4; ++i) {
      const int ni = tet.vertids[i];

      for (int j = 0; j < 4; ++j) {
        const int nj = tet.vertids[j];
        const double m_ij = (i == j) ? mass / 10.0 : mass / 20.0;

        // Add mass contribution for each DOF (x, y, z)
        for (int dof = 0; dof < 3; ++dof) {
          const int row = 3 * ni + dof;
          const int col = 3 * nj + dof;
          triplets.emplace_back(row, col, m_ij);
        }
      }
    }
  }

  mass_matrix_.setFromTriplets(triplets.begin(), triplets.end());
  mass_matrix_.makeCompressed();

  // Add Rayleigh mass damping if specified
  if (material_.alpha_m > 0.0) {
    damping_matrix_ = material_.alpha_m * mass_matrix_;
  }

  std::cout << "[Solid] Mass matrix assembled: " << total_dof << " x "
            << total_dof << " (nnz=" << mass_matrix_.nonZeros() << ")"
            << std::endl;
}

void SolidMechanicsSolver::assembleStiffnessMatrix() {
  if (!mesh_) {
    std::cerr << "[Solid] Error: mesh not initialized" << std::endl;
    return;
  }

  const size_t nv = mesh_->nVertices();
  const size_t nt = mesh_->nTets();
  const size_t total_dof = 3 * nv;

  stiffness_matrix_.resize(total_dof, total_dof);

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(nt * 144);

  // Get material properties
  const double lambda = material_.lambda;
  const double mu = material_.mu;

  // Assemble stiffness matrix element by element
  for (size_t ti = 0; ti < nt; ++ti) {
    const auto &tet = mesh_->tetAt(ti);
    const auto &grads = mesh_->getTetGradients(ti);
    const double vol = tet.volume;

    // For linear elasticity: K = ∫ B^T C B dV
    // where B is the strain-displacement matrix
    // For constant strain tetrahedra, this integral is exact

    for (int i = 0; i < 4; ++i) {
      const int ni = tet.vertids[i];
      const Eigen::Vector3d &grad_i = grads[i];

      for (int j = 0; j < 4; ++j) {
        const int nj = tet.vertids[j];
        const Eigen::Vector3d &grad_j = grads[j];

        // Build element stiffness matrix (3x3 block)
        // K_ij = vol * (λ * (∇Ni)(∇Nj)^T + μ * (∇Ni·∇Nj)*I + μ * (∇Ni)(∇Nj)^T)
        Eigen::Matrix3d k_block;

        // Simpler formulation for constant strain element:
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            k_block(a, b) = vol * (lambda * grad_i(a) * grad_j(b) +
                                   mu * grad_i(b) * grad_j(a));
            if (a == b) {
              k_block(a, b) += vol * mu * grad_i.dot(grad_j);
            }
          }
        }

        // Add to global stiffness matrix
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            const int row = 3 * ni + a;
            const int col = 3 * nj + b;
            triplets.emplace_back(row, col, k_block(a, b));
          }
        }
      }
    }
  }

  stiffness_matrix_.setFromTriplets(triplets.begin(), triplets.end());
  stiffness_matrix_.makeCompressed();

  // Add Rayleigh stiffness damping if specified
  if (material_.alpha_k > 0.0) {
    if (damping_matrix_.size() == 0) {
      damping_matrix_ = material_.alpha_k * stiffness_matrix_;
    } else {
      damping_matrix_ += material_.alpha_k * stiffness_matrix_;
    }
  }

  std::cout << "[Solid] Stiffness matrix assembled: " << total_dof << " x "
            << total_dof << " (nnz=" << stiffness_matrix_.nonZeros() << ")"
            << std::endl;
}

bool SolidMechanicsSolver::solveStatic() {
  if (!is_initialized_) {
    std::cerr << "[Solid] Error: solver not initialized" << std::endl;
    return false;
  }

  const size_t nv = mesh_->nVertices();
  const size_t total_dof = 3 * nv;

  // Build force vector: f = body_force + surface_traction + fsi_traction
  Eigen::VectorXd force_vector = Eigen::VectorXd::Zero(total_dof);

  for (size_t i = 0; i < nv; ++i) {
    Eigen::Vector3d total_force = body_force_[i] + surface_traction_[i];
    if (!fsi_traction_.empty()) {
      total_force += fsi_traction_[i];
    }
    force_vector(3 * i + 0) = total_force(0);
    force_vector(3 * i + 1) = total_force(1);
    force_vector(3 * i + 2) = total_force(2);
  }

  // Apply boundary conditions: K u = f
  Eigen::SparseMatrix<double> K_bc = stiffness_matrix_;
  Eigen::VectorXd f_bc = force_vector;
  const double large_val = 1e20;

  // Enforce fixed displacement BCs
  for (size_t i = 0; i < nv; ++i) {
    if (mesh_->getSolidVertexBC(i) == SolidBCType::Fixed) {
      for (int dof = 0; dof < 3; ++dof) {
        long int idx = 3 * i + dof;
        // Modify matrix using large diagonal penalty method
        for (int k = 0; k < K_bc.outerSize(); ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(K_bc, k); it;
               ++it) {
            if (it.row() == idx) {
              if (it.col() == idx) {
                it.valueRef() = large_val; // Large diagonal
              } else {
                it.valueRef() = 0.0;
              }
            } else if (it.col() == idx) {
              it.valueRef() = 0.0;
            }
          }
        }
        f_bc(idx) = 0.0; // Fixed displacement = 0
      }
    }
  }

  // Solve using sparse direct solver
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(K_bc);

  if (solver.info() != Eigen::Success) {
    std::cerr << "[Solid] LU decomposition failed" << std::endl;
    return false;
  }

  Eigen::VectorXd u_sol = solver.solve(f_bc);

  if (solver.info() != Eigen::Success) {
    std::cerr << "[Solid] Linear solve failed" << std::endl;
    return false;
  }

  // Extract displacement
  for (size_t i = 0; i < nv; ++i) {
    displacement_[i](0) = u_sol(3 * i + 0);
    displacement_[i](1) = u_sol(3 * i + 1);
    displacement_[i](2) = u_sol(3 * i + 2);
    total_displacement_[i] += displacement_[i];
    displaced_positions_[i] = original_positions_[i] + total_displacement_[i];
  }

  // Compute strain and stress
  computeStrain();
  computeStress();
  computeVonMises();

  std::cout << "[Solid] Static solve completed" << std::endl;
  return true;
}

bool SolidMechanicsSolver::solveDynamicStep() {
  if (!is_initialized_) {
    std::cerr << "[Solid] Error: solver not initialized" << std::endl;
    return false;
  }

  // Newmark-beta time integration (implicit)
  // Standard formulation:
  // M*a_{n+1} + C*v_{n+1} + K*u_{n+1} = F_{n+1}
  // u_{n+1} = u_n + dt*v_n + dt²*((0.5-β)*a_n + β*a_{n+1})
  // v_{n+1} = v_n + dt*((1-γ)*a_n + γ*a_{n+1})
  //
  // Substituting into equilibrium equation gives:
  // (M/(β*dt²) + γ*C/(β*dt) + K)*u_{n+1} = F + M*(...) + C*(...)

  const size_t nv = mesh_->nVertices();
  const size_t total_dof = 3 * nv;

  // Build effective stiffness matrix: K_eff = K + M/(β*dt²) + γ*C/(β*dt)
  Eigen::SparseMatrix<double> K_eff =
      stiffness_matrix_ + mass_matrix_ / (beta_ * dt_ * dt_);

  if (damping_matrix_.size() > 0) {
    K_eff += damping_matrix_ * gamma_ / (beta_ * dt_);
  }

  // Build effective force vector
  Eigen::VectorXd force_vector = Eigen::VectorXd::Zero(total_dof);

  // External forces (gravity, surface traction, FSI)
  for (size_t i = 0; i < nv; ++i) {
    Eigen::Vector3d total_force = body_force_[i] + surface_traction_[i];
    if (!fsi_traction_.empty()) {
      total_force += fsi_traction_[i];
    }

    for (int dof = 0; dof < 3; ++dof) {
      force_vector(3 * i + dof) = total_force(dof);
    }
  }

  // Add inertial terms from mass matrix:
  // M * (u_n/(β*dt²) + v_n/(β*dt) + (1/(2β)-1)*a_n)
  Eigen::VectorXd u_prev_vec(total_dof);
  Eigen::VectorXd v_prev_vec(total_dof);
  Eigen::VectorXd a_prev_vec(total_dof);

  for (size_t i = 0; i < nv; ++i) {
    for (int dof = 0; dof < 3; ++dof) {
      u_prev_vec(3 * i + dof) = displacement_prev_[i](dof);
      v_prev_vec(3 * i + dof) = velocity_prev[i](dof);
      a_prev_vec(3 * i + dof) = acceleration_prev_[i](dof);
    }
  }

  force_vector += mass_matrix_ * (u_prev_vec / (beta_ * dt_ * dt_) +
                                  v_prev_vec / (beta_ * dt_) +
                                  a_prev_vec * (1.0 / (2.0 * beta_) - 1.0));

  // Add damping terms if present:
  // C * (γ*u_n/(β*dt) + (γ/(β) - 1)*v_n + dt*(γ/(2β) - 1)*a_n)
  if (damping_matrix_.size() > 0) {
    force_vector +=
        damping_matrix_ * (u_prev_vec * gamma_ / (beta_ * dt_) +
                           v_prev_vec * (gamma_ / beta_ - 1.0) +
                           a_prev_vec * dt_ * (gamma_ / (2.0 * beta_) - 1.0));
  }

  // Apply boundary conditions
  const double large_val = 1e20;
  for (size_t i = 0; i < nv; ++i) {
    if (mesh_->getSolidVertexBC(i) == SolidBCType::Fixed) {
      for (int dof = 0; dof < 3; ++dof) {
        long int idx = 3 * i + dof;
        for (int k = 0; k < K_eff.outerSize(); ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(K_eff, k); it;
               ++it) {
            if (it.row() == idx) {
              if (it.col() == idx) {
                it.valueRef() = large_val;
              } else {
                it.valueRef() = 0.0;
              }
            } else if (it.col() == idx) {
              it.valueRef() = 0.0;
            }
          }
        }
        force_vector(idx) = 0.0;
      }
    }
  }

  // Solve system
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(K_eff);

  if (solver.info() != Eigen::Success) {
    std::cerr << "[Solid] Dynamic solve: LU decomposition failed" << std::endl;
    return false;
  }

  Eigen::VectorXd u_sol = solver.solve(force_vector);

  if (solver.info() != Eigen::Success) {
    std::cerr << "[Solid] Dynamic solve: linear solve failed" << std::endl;
    return false;
  }

  // Extract displacement solution
  for (size_t i = 0; i < nv; ++i) {
    displacement_[i](0) = u_sol(3 * i + 0);
    displacement_[i](1) = u_sol(3 * i + 1);
    displacement_[i](2) = u_sol(3 * i + 2);
  }

  // Update acceleration using Newmark formula:
  // a_{n+1} = (u_{n+1} - u_n - dt*v_n) / (β*dt²) - (1-2β)/(2β)*a_n
  for (size_t i = 0; i < nv; ++i) {
    acceleration_[i] =
        (displacement_[i] - displacement_prev_[i] - dt_ * velocity_prev[i]) /
            (beta_ * dt_ * dt_) -
        (1.0 - 2.0 * beta_) / (2.0 * beta_) * acceleration_prev_[i];
  }

  // Update velocity using Newmark formula:
  // v_{n+1} = v_n + dt*((1-γ)*a_n + γ*a_{n+1})
  for (size_t i = 0; i < nv; ++i) {
    velocity_[i] =
        velocity_prev[i] + dt_ * ((1.0 - gamma_) * acceleration_prev_[i] +
                                  gamma_ * acceleration_[i]);
  }

  // Update total displacement (cumulative)
  for (size_t i = 0; i < nv; ++i) {
    total_displacement_[i] += displacement_[i] - displacement_prev_[i];
    displaced_positions_[i] = original_positions_[i] + total_displacement_[i];
  }

  // Update previous values for next step
  displacement_prev_ = displacement_;
  velocity_prev = velocity_;
  acceleration_prev_ = acceleration_;

  // Update time
  time_ += dt_;

  // Compute strain and stress
  computeStrain();
  computeStress();
  computeVonMises();

  return true;
}

void SolidMechanicsSolver::computeStrain() {
  if (!mesh_)
    return;

  // Initialize strain tensors
  for (auto &s : strain_tensor_) {
    s = Eigen::Matrix3d::Zero();
  }

  // Compute strain at each element (constant strain tetrahedra)
  for (size_t ti = 0; ti < mesh_->nTets(); ++ti) {
    const auto &tet = mesh_->tetAt(ti);
    const auto &grads = mesh_->getTetGradients(ti);

    // Displacement gradient: F_ij = du_i/dx_j
    Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 4; ++i) {
      const int vi = tet.vertids[i];
      const Eigen::Vector3d &grad = grads[i];
      const Eigen::Vector3d &u = displacement_[vi];

      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          F(a, b) += u(a) * grad(b);
        }
      }
    }

    // Small strain tensor: ε = 0.5 * (F + F^T)
    Eigen::Matrix3d epsilon = 0.5 * (F + F.transpose());

    // Assign to vertices (average)
    for (int i = 0; i < 4; ++i) {
      strain_tensor_[tet.vertids[i]] += epsilon / 4.0;
    }
  }
}

void SolidMechanicsSolver::computeStress() {
  if (!mesh_)
    return;

  const size_t nv = mesh_->nVertices();
  const double lambda = material_.lambda;
  const double mu = material_.mu;

  // Stress-strain relationship: σ = λ*tr(ε)*I + 2*μ*ε
  for (size_t i = 0; i < nv; ++i) {
    const Eigen::Matrix3d &eps = strain_tensor_[i];
    double trace = eps.trace();
    stress_tensor_[i] =
        lambda * trace * Eigen::Matrix3d::Identity() + 2.0 * mu * eps;
  }
}

void SolidMechanicsSolver::computeVonMises() {
  const size_t nv = mesh_->nVertices();

  for (size_t i = 0; i < nv; ++i) {
    const Eigen::Matrix3d &s = stress_tensor_[i];

    // Von Mises stress: sqrt(3/2 * s':s') where s' is deviatoric stress
    double mean_stress = s.trace() / 3.0;
    Eigen::Matrix3d s_dev = s - mean_stress * Eigen::Matrix3d::Identity();

    double vm = std::sqrt(1.5 * (s_dev.array() * s_dev.array()).sum());
    von_mises_stress_[i] = vm;
  }
}

const std::vector<Eigen::Vector3d> &
SolidMechanicsSolver::getTotalDisplacement() const {
  return total_displacement_;
}

const std::vector<Eigen::Vector3d> &
SolidMechanicsSolver::getDisplacedPositions() const {
  return displaced_positions_;
}

const std::vector<Eigen::Vector3d> &SolidMechanicsSolver::getVlocity() const {
  return velocity_;
}

void SolidMechanicsSolver::setFsiTraction(
    const std::vector<Eigen::Vector3d> &traction) {
  fsi_traction_ = traction;
}

std::tuple<double, double, double> SolidMechanicsSolver::get_stats() const {
  double max_disp = 0.0;
  double max_vm = 0.0;
  double max_vel = 0.0;

  for (size_t i = 0; i < displacement_.size(); ++i) {
    max_disp = std::max(max_disp, displacement_[i].norm());
    max_vel = std::max(max_vel, velocity_[i].norm());
  }

  for (double vm : von_mises_stress_) {
    max_vm = std::max(max_vm, vm);
  }

  return {max_disp, max_vm, max_vel};
}
