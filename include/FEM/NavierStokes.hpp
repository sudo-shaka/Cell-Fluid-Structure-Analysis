#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <memory>
#include <vector>

// Forward declarations to avoid circular include with Mesh.hpp
class Mesh;
struct Face;

enum class FluidBCType {
  Wall, // noSlip
  Inlet,
  Outlet,
  Internal,
  Undefined,
};

enum class OutletType {
  DirichletPressure,
  Neumann,
  Undefined,
};

enum class InletType {
  Uniform,
  Pulsitile,
  Undefined,
};

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
  int reference_node_ = -1;

  double tolerance_ = 1e-12;
  int n_corrections_ = 1;
  double relax_u_ = 1.0;
  double relax_p_ = 1.0;
  double dt_ = 1e-4;
  double time_ = 0.0;

  Fluid fluid_properties_;

  std::vector<Eigen::Vector3d> velocity_;
  std::vector<Eigen::Vector3d> velocity_star_;
  std::vector<double> pressure_;
  std::vector<double> pressure_correction_;
  std::vector<double> inv_lumped_mass_;

  Eigen::Vector3d mean_inlet_velocity_;
  InletType inlet_type_;
  double outlet_pressure_ = 0.0;
  OutletType outlet_type_;

  std::shared_ptr<Mesh> mesh_ptr_ = nullptr;
  std::unique_ptr<Eigen::SparseMatrix<double>> mass_matrix_;
  std::unique_ptr<Eigen::SparseMatrix<double>> gradient_matrix_x_;
  std::unique_ptr<Eigen::SparseMatrix<double>> gradient_matrix_y_;
  std::unique_ptr<Eigen::SparseMatrix<double>> gradient_matrix_z_;
  std::unique_ptr<Eigen::SparseMatrix<double>> stiffness_matrix_;
  std::unique_ptr<Eigen::SparseMatrix<double>> poisson_matrix_;

  void reenforceVelocityBCs();
  void reinforcePressureBCs();

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

  bool pisoStep();
  bool solveMomentumPredictor();
  void computeDivergence(const std::vector<Eigen::Vector3d> &u,
                         std::vector<double> &div_out);
  std::vector<Eigen::Vector3d> computeAdvectionRHS();
  bool correctVelocity();
  bool solvePressurePoisson();
  bool solvePressureCG(const std::vector<double> &rhs, std::vector<double> &x);
  bool solvePressureBiCGSTAB(const std::vector<double> &rhs,
                             std::vector<double> &x);
  bool normalizePressire();

  bool hasNans() {
    for (const auto &v : velocity_) {
      if (v.array().isNaN().any()) {
        return true;
      }
    }
    for (const auto &v : velocity_star_) {
      if (v.array().isNaN().any()) {
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
  std::vector<Eigen::Vector3d> computePressureForces() const;
  std::vector<Eigen::Vector3d> computeShearStress() const;
  std::vector<Eigen::Vector3d> computeTotalFluidForces() const;
  Eigen::Vector3d computeTotalMomentum() const;
  double computeNetBoundaryFlux() const;

  // getters
  double getViscosity() const { return fluid_properties_.viscosity; }
  double getDensity() const { return fluid_properties_.density; }
  double getTolerance() const { return tolerance_; }
  const std::vector<double> &getEffectiveViscosity() const {
    return fluid_properties_.effective_viscosity;
  }
  double getPressureAtNode(size_t ni) const {
    assert(ni < pressure_.size());
    return pressure_[ni];
  }
  const Eigen::Vector3d &getVelocityAtNode(size_t ni) const {
    assert(ni < velocity_.size());
    return velocity_[ni];
  }
  const Mesh &getMesh() const { return *mesh_ptr_; }

  // setters for boundary conditions / references
  double getMeanFacePressure(size_t fi) const;
  Eigen::Vector3d getPressureForceAtFace(size_t fi) const;

  // setters
  void setMeanInletVelocity(const Eigen::Vector3d &vel) {
    mean_inlet_velocity_ = vel;
  }
  void setOutletType(OutletType type) { outlet_type_ = type; }
  void setOutletPressure(double p) {
    if (outlet_type_ != OutletType::DirichletPressure) {
      std::cout << "[NS Solver] Warning: setting outlet pressure without "
                   "dirichlet outlet being the outlet type"
                << std::endl;
    }
    outlet_pressure_ = p;
  }
  void setInletType(InletType type) { inlet_type_ = type; }
  void setNumberOfCorrections(int n) { n_corrections_ = n; }
  void setRelaxU(double relax) { relax_u_ = relax; }
  void setRelaxP(double relax) { relax_p_ = relax; }
  void setReferencePressureNode(int node) { reference_node_ = node; }
  void setDt(double dt) { dt_ = dt; }
  double getDt() const { return dt_; }
  void setTolerance(double tol) { tolerance_ = tol; }
  double getTime() const { return time_; }
  const std::vector<Eigen::Vector3d> &getVelocity() const { return velocity_; }
  const std::vector<double> &getPressure() const { return pressure_; }

  // Simple solver method: basic operator splitting without PISO iteration
  bool simpleStep();
};

/**
 * @brief Simple Navier-Stokes solver using basic operator splitting
 *
 * Alternative to PISO for quick prototyping and testing.
 * Uses single momentum prediction and one pressure correction.
 */
class NavierStokesSimpleSolver {
public:
  explicit NavierStokesSimpleSolver(
      std::shared_ptr<NavierStokesSolver> ns_solver)
      : solver_(ns_solver) {}

  /**
   * @brief Perform a single time step with simple operator splitting
   * @return true if step succeeded, false otherwise
   */
  bool step() {
    if (!solver_) {
      return false;
    }
    return solver_->simpleStep();
  }

  /**
   * @brief Advance multiple time steps
   * @param n Number of steps to advance
   * @return true if all steps succeeded
   */
  bool advanceSteps(int n) {
    for (int i = 0; i < n; ++i) {
      if (!step()) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Run simulation until specified time
   * @param target_time Target time to reach
   * @return true if target time was reached successfully
   */
  bool runUntil(double target_time) {
    while (getSolver()->getTime() < target_time) {
      if (!step()) {
        return false;
      }
    }
    return true;
  }

  double getTime() const { return solver_->getTime(); }
  std::shared_ptr<NavierStokesSolver> getSolver() { return solver_; }

private:
  std::shared_ptr<NavierStokesSolver> solver_;
};
