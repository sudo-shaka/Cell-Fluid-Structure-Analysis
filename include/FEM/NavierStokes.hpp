#pragma once

#include "BC/BC.hpp"
#include "LinearAlgebra/LinearSolvers.hpp"
#include "LinearAlgebra/SparseMatrix.hpp"
#include "Mesh/Mesh.hpp"
#include <glm/glm.hpp>
#include <memory>

class NavierStokesSolver {
  bool is_initialized_ = false;
  int reference_node = -1;

  double relax_u = 1.0;
  double relax_p = 1.0;
  double sor_relaxation = 1.0; // omega
  double dt_;
  double time_;

  double density_;
  double viscosity_;
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
  void buildGradientMatrices();
  void buildMessMatrix();
  void buildStiffnessMatrix();
  void buildPoissonMatrix();
  void build_sparse_matrices() {
    buildGradientMatrices();
    buildMessMatrix();
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
};
