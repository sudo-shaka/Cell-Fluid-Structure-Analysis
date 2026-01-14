#include "BC/BC.hpp"
#include "FEM/SolidMechanics.hpp"
#include "LinearAlgebra/LinearSolvers.hpp"
#include <glm/matrix.hpp>
#include <memory>
#include <cassert>

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
      displacement_smoothing_factor_(0.3), max_displacement_factor_(0.1) {
  // Configure linear solver defaults
  linear_solver_.setMaxCorrections(max_iter_);
  linear_solver_.setTolerance(tolerance_);
  linear_solver_.setPreconditoner(Preconditioner());
}

void SolidMechanicsSolver::initialize(std::shared_ptr<Mesh> mesh_ptr,
                                      const Material &mat) {
  std::cout << "[Solid] Initializing solver." << std::endl;
  mesh_ = mesh_ptr;
  material_ = mat;
  original_positions_ = mesh_->getVertPositions();

  size_t nv = mesh_->nVertices();
  displacement_.resize(nv, glm::dvec3{0.0});
  velocity_.resize(nv, glm::dvec3{0.0});
  acceleration_.resize(nv, glm::dvec3{0.0});
  displacement_prev_.resize(nv, glm::dvec3{0.0});
  velocity_prev.resize(nv, glm::dvec3{0.0});
  acceleration_prev_.resize(nv, glm::dvec3{0.0});
  total_displacement_.resize(nv, glm::dvec3{0.0});
  strain_tensor_.resize(nv);
  stress_tensor_.resize(nv);
  von_mises_stress_.resize(nv, 0.0);
  principle_stresses_.resize(nv, glm::dvec3{0.0});
  body_force_.resize(nv, glm::dvec3{0.0});
  surface_traction_.resize(nv, glm::dvec3{0.0});
  assembleMassMatrix();
  assembleStiffnessMatrix();
  is_initialized_ = true;
  std::cout << "[Solid] Initialization complete." << std::endl;
}

void SolidMechanicsSolver::applyGravity(glm::dvec3 g) {
  if (!mesh_)
    return;

  for (const auto &tet : mesh_->getTets()) {
    double vol = tet.volume;
    double nodal_mass = material_.density * vol / 4.0; // Lumped mass per node

    for (const auto &i : tet.vertids) {
      int v_id = tet.vertids[i];
      body_force_[v_id] += nodal_mass * g;
    }
  }
}

void SolidMechanicsSolver::applyPressure(double pressure) {
  assert(mesh);
  assert(pressures.size() == surface_traction_.size());
  const auto &faces = mesh_->getFaces();
  for (const auto &f : faces) {
    if (!f.is_ecm)
      continue;
    glm::dvec3 traction = pressure * f.normal;
    glm::dvec3 nodal_force = traction * f.area / 3.0;

    for (const auto &i : f.vertids) {
      if (mesh_->getSolidVertexBC(i) == SolidBCType::Fixed)
        continue;
      surface_traction_[i] += nodal_force;
    }
  }
}
void SolidMechanicsSolver::applyPressure(const std::vector<double> &pressures) {
  assert(pressures.size() == surface_traction_.size());
  assert(mesh);
  size_t nv = mesh_->nVertices();
  std::vector<glm::dvec3> global_forces(nv, glm::dvec3{0.0});
  for (const auto &f : mesh_->getFaces()) {
    if (!f.is_ecm)
      continue;
    for (const auto &i : f.vertids) {
      if (mesh_->getSolidVertexBC(i) == SolidBCType::Fixed)
        continue;
      glm::dvec3 traction = pressures[i] * f.normal;
      global_forces[i] +=
          traction * f.area / static_cast<double>(f.vertids.size());
    }
  }
  for (size_t vi = 0; vi < mesh_->nVertices(); vi++)
    surface_traction_[vi] += global_forces[vi];
}

void SolidMechanicsSolver::addBodyForce(size_t node_id,
                                        const glm::dvec3 &force) {
  assert(node_id < body_force_.size());
  body_force_[node_id] += force;
}

void SolidMechanicsSolver::assembleMassMatrix() {
  if (!mesh_)
    return;
  std::vector<std::tuple<int, int, double>> triplets;
  const auto &tets = mesh_->getTets();
  for (const auto &tet : tets) {
    double diag = material_.density * tet.volume / 10.0;
    double offdiag = material_.density * tet.volume / 20.0;
    for (int i = 0; i < 4; ++i) {
      int gi = tet.vertids[i];
      for (int j = 0; j < 4; ++j) {
        int gj = tet.vertids[j];
        double val = (i == j) ? diag : offdiag;
        triplets.emplace_back(gi, gj, val);
      }
    }
  }
  const size_t nv = mesh_->nVertices();
  SparseMatrix::buildCsrFromTriplets(nv, triplets, mass_matrix_, true);
}

void SolidMechanicsSolver::assembleStiffnessMatrix() {
  if (!mesh_)
    return;

  std::vector<std::tuple<int, int, double>> triplets;
  const auto &tets = mesh_->getTets();

  for (size_t e = 0; e < tets.size(); ++e) {
    const Tet &tet = tets[e];
    const auto &grad_n = mesh_->getTetGradient(e);

    // B matrix relates strain_ to nodal displacements
    // For each pair of nodes, compute stiffness contribution
    for (int i = 0; i < 4; ++i) {
      int gi = tet.vertids[i];
      for (int j = 0; j < 4; ++j) {
        int gj = tet.vertids[j];

        // For isotropic linear elasticity with same stiffness in all
        // directions:
        double lambda = material_.lambda;
        double mu = material_.mu;

        // Contribution from divergence terms (lambda)
        double div_contrib =
            lambda * tet.volume * glm::dot(grad_n[i], grad_n[j]);

        // Contribution from gradient terms (2*mu)
        double grad_contrib =
            2.0 * mu * tet.volume * glm::dot(grad_n[i], grad_n[j]);

        double val = div_contrib + grad_contrib;
        triplets.emplace_back(gi, gj, val);
      }
    }
  }

  SparseMatrix::buildCsrFromTriplets(mesh_->nVertices(), triplets,
                                     stiffness_matrix_, true);
}
