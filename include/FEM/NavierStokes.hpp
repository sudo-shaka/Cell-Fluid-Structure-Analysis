#pragma once

#include "BC/BC.hpp"
#include "LinearAlgebra/LinearSolvers.hpp"
#include "LinearAlgebra/SparseMatrix.hpp"
#include "Mesh/Mesh.hpp"
#include <glm/glm.hpp>
#include <memory>

enum class ViscosityModel {
  Newtonian,
  Carreau, // TODO
};

enum class TurbulenceModel {
  Laminar, // default - no turbulence
  // TODO:
  RANS, // Reynolds
  LES,  // Large Eddy
};

struct Fluid {
  double density = 1000;    // water-like
  double viscosity = 0.001; // water-like
  TurbulenceModel turbuelence_model = TurbulenceModel::Laminar;
  ViscosityModel viscosity_model = ViscosityModel::Newtonian;
  std::vector<double> effective_viscosity;
};

class NavierStokesSolver {
  bool is_initialized_ = false;
  int reference_node = -1;

  Fluid fluid_properties_;

  double relax_u = 1.0;
  double relax_p = 1.0;
  double sor_relaxation = 1.0; // omega
  double dt_ = 1e-4;
  double time_ = 0.0;
  double tolerance = 1e-6;

  std::vector<glm::dvec3> velocity_;
  std::vector<glm::dvec3> velocity_star_;
  std::vector<double> pressure_;
  std::vector<double> pressure_correction_;

  glm::dvec3 mean_inlet_velocity;
  InletType inlet_type_;
  OutletType outlet_type_;

  std::shared_ptr<Mesh> mesh_ptr_ = nullptr;
  std::unique_ptr<SparseMatrix> mass_matrix_;
  std::unique_ptr<SparseMatrix> gradient_matrix_x_;
  std::unique_ptr<SparseMatrix> gradient_matrix_y_;
  std::unique_ptr<SparseMatrix> gradient_matrix_z_;
  std::unique_ptr<SparseMatrix> stiffness_matrix_;
  std::unique_ptr<SparseMatrix> poisson_matrix_;
  Preconditioner poisson_preconditioner_;

public:
  NavierStokesSolver() = default;

  void initialize(std::shared_ptr<Mesh> mesh_ptr,
                  const Fluid &fluid_properties);
  void buildGradientMatrices();
  void buildMassMatrix();
  void buildStiffnessMatrix();
  void buildPoissonMatrix();
  void build_sparse_matrices() {
    buildGradientMatrices();
    buildMassMatrix();
    buildPoissonMatrix();
    buildStiffnessMatrix();
  }

  bool pisoStep(size_t n_iter, double dt);
  bool solveMomentumPredictor();
  bool correctVelocity();
  bool solvePressurePoisson();
  bool solvePressureCG(const std::vector<double> &rhs, std::vector<double> &x);
  bool solvePressureBiCGSTAB(const std::vector<double> &rhs,
                             std::vector<double> &x);
  bool normalizePressire();
  friend void
  boundary_assignment::applyBoundaryConditions(NavierStokesSolver &solver);

  bool hasNans() {
    for (const auto &v : velocity_) {
      if (glm::any(glm::isnan(v))) {
        return true;
      }
    }
    for (const auto &v : velocity_star_) {
      if (glm::any(glm::isnan(v))) {
        return true;
      }
    }
    for (const auto &p : pressure_) {
      if (std::isnan(p)) {
        return true;
      }
    }
    for (const auto &p : pressure_correction_) {
      if (std::isnan(p)) {
        return true;
      }
    }
    return false;
  }
  std::vector<glm::dvec3> computePressureForces() const;
  std::vector<glm::dvec3> computeShearStress() const;
  glm::dvec3 computeTotalMomentum() const;
  double computeNetBoundaryFlux() const;

  // getters
  double getViscosity() const { return fluid_properties_.viscosity; }
  double getDensity() const { return fluid_properties_.density; }
  const std::vector<double> &getEffectiveViscosity() const {
    return fluid_properties_.effective_viscosity;
  }
  double getPressureAtNode(size_t ni) const {
    assert(ni < pressure_.size());
    return pressure_[ni];
  }
  const glm::dvec3 &getVelocityAtNode(size_t ni) const {
    assert(nv < velocity_.size());
    return velocity_[ni];
  }
  double getMeanFacePressure(size_t fi) const {
    assert(fi < mesh_ptr_->nFaces());
    const auto &faces = mesh_ptr_->getFaces();
    double average_pressure = 0.0;
    for (const auto &vid : faces[fi].vertids) {
      average_pressure += pressure_[vid];
    }
    return average_pressure / static_cast<double>(faces[fi].vertids.size());
  }
  glm::dvec3 getPressureForceAtFace(size_t fi) const {
    assert(fi < mesh_ptr_->nFaces());
    double pressure = getMeanFacePressure(fi);
    const Face &face = mesh_ptr_->getFaces()[fi];
    return -face.normal * face.area * pressure;
  }
};
