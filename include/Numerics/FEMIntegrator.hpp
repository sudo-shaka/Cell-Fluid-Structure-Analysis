#pragma once

#include "FEM/NavierStokes.hpp"
#include "FEM/SolidMechanics.hpp"
#include "Mesh/Mesh.hpp"
#include <algorithm>
#include <iostream>
#include <memory>

/**
 * @brief Time integrator for Navier-Stokes solver using PISO algorithm
 */
class NavierStokesIntegrator {
public:
  NavierStokesIntegrator(std::shared_ptr<NavierStokesSolver> solver)
      : solver_(solver), time_(0.0), step_count_(0) {}

  /**
   * @brief Advance one time step using PISO algorithm
   * @return true if step succeeded
   */
  bool step() {
    if (!solver_) {
      return false;
    }

    bool success = solver_->pisoStep();
    if (success) {
      time_ = solver_->getTime();
      step_count_++;

      // CFL diagnostics
      /*
      const double dt = solver_->getDt();
      const auto &mesh = solver_->getMesh();
      const double h_min = mesh.getMinEdgeLength();
      const double nu = solver_->getViscosity();
      double u_max = 0.0;
      for (const auto &v : solver_->getVelocity()) {
        u_max = std::max(u_max, v.norm());
      }
      const double cfl_conv = (h_min > 0.0) ? (dt * u_max / h_min) : 0.0;
      const double cfl_diff = (h_min > 0.0) ? (dt * nu / (h_min * h_min)) : 0.0;
      std::cout << "[CFL] h_min=" << h_min << ", u_max=" << u_max
                << ", convective=" << cfl_conv
                << ", diffusive=" << cfl_diff << std::endl;

      // Recommend dt limits for target CFL=0.5
      const double target = 0.5;
      const double dt_conv_lim = (u_max > 0.0) ? (target * h_min / u_max) : dt;
      const double dt_diff_lim = (nu > 0.0) ? (target * h_min * h_min / nu) : dt;
      std::cout << "[CFL] dt limits: conv<=" << dt_conv_lim
                << ", diff<=" << dt_diff_lim << std::endl;
                */
    }
    return success;
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
    while (time_ < target_time) {
      if (!step()) {
        return false;
      }
    }
    return true;
  }

  double getTime() const { return time_; }
  int getStepCount() const { return step_count_; }
  std::shared_ptr<NavierStokesSolver> getSolver() { return solver_; }

private:
  std::shared_ptr<NavierStokesSolver> solver_;
  double time_;
  int step_count_;
};

/**
 * @brief Time integrator for solid mechanics solver
 */
class SolidMechanicsIntegrator {
public:
  enum class IntegrationType { Static, Dynamic };

  SolidMechanicsIntegrator(std::shared_ptr<SolidMechanicsSolver> solver,
                           IntegrationType type = IntegrationType::Dynamic)
      : solver_(solver), type_(type), time_(0.0), step_count_(0) {}

  /**
   * @brief Advance one time step
   * @return true if step succeeded
   */
  bool step() {
    if (!solver_) {
      return false;
    }

    bool success;
    if (type_ == IntegrationType::Static) {
      success = solver_->solveStatic();
    } else {
      success = solver_->solveDynamicStep();
      if (success) {
        time_ += solver_->getDt();
      }
    }

    if (success) {
      step_count_++;
    }
    return success;
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
   * @brief Run simulation until specified time (dynamic only)
   * @param target_time Target time to reach
   * @return true if target time was reached successfully
   */
  bool runUntil(double target_time) {
    if (type_ == IntegrationType::Static) {
      return step(); // Static analysis is single step
    }

    while (time_ < target_time) {
      if (!step()) {
        return false;
      }
    }
    return true;
  }

  double getTime() const { return time_; }
  int getStepCount() const { return step_count_; }
  std::shared_ptr<SolidMechanicsSolver> getSolver() { return solver_; }
  IntegrationType getType() const { return type_; }

private:
  std::shared_ptr<SolidMechanicsSolver> solver_;
  IntegrationType type_;
  double time_;
  int step_count_;
};

/**
 * @brief Coupled Fluid-Structure Interaction (FSI) integrator
 *
 * This integrator couples Navier-Stokes and Solid Mechanics solvers
 * for two-way FSI simulation. The solvers share the same mesh.
 *
 * Coupling scheme:
 * 1. Fluid solver advances one time step on current mesh
 * 2. Compute pressure forces from fluid on structure
 * 3. Apply pressure forces to solid solver
 * 4. Solid solver advances one time step
 * 5. Deform mesh based on solid displacement
 * 6. Rebuild fluid matrices on deformed mesh
 * 7. Repeat
 */
class FSICoupledIntegrator {
public:
  enum class CouplingScheme {
    /// Weakly coupled (explicit): fluid -> solid -> mesh update in sequence
    Explicit,
    /// Strongly coupled (implicit): iterate until convergence within each time
    /// step
    Implicit
  };

  /**
   * @brief Construct FSI coupled integrator
   * @param fluid_solver Navier-Stokes solver (must share mesh with
   * solid_solver)
   * @param solid_solver Solid mechanics solver (must share mesh with
   * fluid_solver)
   * @param scheme Coupling scheme (default: Explicit)
   */
  FSICoupledIntegrator(std::shared_ptr<NavierStokesSolver> fluid_solver,
                       std::shared_ptr<SolidMechanicsSolver> solid_solver,
                       CouplingScheme scheme = CouplingScheme::Explicit)
      : fluid_solver_(fluid_solver), solid_solver_(solid_solver),
        scheme_(scheme), time_(0.0), step_count_(0), max_fsi_iterations_(10),
        fsi_tolerance_(1e-4), enable_mesh_update_(true),
        enable_matrix_rebuild_(true) {

    // Verify both solvers share the same mesh
    if (!fluid_solver_ || !solid_solver_) {
      std::cerr << "[FSI] Error: Solvers not initialized" << std::endl;
      return;
    }

    // Get shared mesh pointer (assumes both use same mesh)
    mesh_ptr_ = solid_solver_->getMeshPtr();

    std::cout << "[FSI] Coupled integrator initialized with "
              << (scheme_ == CouplingScheme::Explicit ? "explicit"
                                                      : "implicit")
              << " coupling" << std::endl;
  }

  /**
   * @brief Advance one coupled FSI time step
   * @return true if step succeeded
   */
  bool step() {
    if (!fluid_solver_ || !solid_solver_ || !mesh_ptr_) {
      std::cerr << "[FSI] Error: Solvers or mesh not initialized" << std::endl;
      return false;
    }

    bool success = false;

    if (scheme_ == CouplingScheme::Explicit) {
      success = stepExplicit();
    } else {
      success = stepImplicit();
    }

    if (success) {
      time_ = std::max(fluid_solver_->getTime(), solid_solver_->getTime());
      step_count_++;
    }

    return success;
  }

  /**
   * @brief Advance multiple coupled time steps
   * @param n Number of steps to advance
   * @return true if all steps succeeded
   */
  bool advanceSteps(int n) {
    for (int i = 0; i < n; ++i) {
      std::cout << "\n[FSI] ========== Step " << (step_count_ + 1)
                << " ==========" << std::endl;
      if (!step()) {
        std::cerr << "[FSI] Step " << (step_count_ + 1) << " failed"
                  << std::endl;
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
    while (time_ < target_time) {
      if (!step()) {
        return false;
      }
    }
    return true;
  }

  // Getters
  double getTime() const { return time_; }
  int getStepCount() const { return step_count_; }
  std::shared_ptr<NavierStokesSolver> getFluidSolver() {
    return fluid_solver_;
  }
  std::shared_ptr<SolidMechanicsSolver> getSolidSolver() {
    return solid_solver_;
  }
  std::shared_ptr<Mesh> getMesh() { return mesh_ptr_; }

  // Setters for coupling parameters
  void setMaxFSIIterations(size_t max_iter) { max_fsi_iterations_ = max_iter; }
  void setFSITolerance(double tol) { fsi_tolerance_ = tol; }
  void enableMeshUpdate(bool enable) { enable_mesh_update_ = enable; }
  void enableMatrixRebuild(bool enable) { enable_matrix_rebuild_ = enable; }

private:
  /**
   * @brief Explicit (weakly coupled) FSI step
   *
   * Algorithm:
   * 1. Solve fluid on current mesh configuration
   * 2. Extract pressure forces from fluid
   * 3. Apply forces to solid and solve
   * 4. Update mesh geometry based on solid displacement
   * 5. Rebuild fluid matrices on new mesh
   */
  bool stepExplicit() {
    // Step 1: Advance fluid solver
    std::cout << "[FSI] Solving fluid..." << std::endl;
    if (!fluid_solver_->pisoStep()) {
      std::cerr << "[FSI] Fluid step failed" << std::endl;
      return false;
    }

    // Step 2: Compute total fluid forces on structure (pressure + viscous shear)
    std::cout << "[FSI] Computing fluid forces (pressure + shear)..." << std::endl;
    std::vector<Eigen::Vector3d> total_forces =
        fluid_solver_->computeTotalFluidForces();

    // Step 3: Apply FSI traction to solid solver
    solid_solver_->setFsiTraction(total_forces);

    // Step 4: Advance solid solver
    std::cout << "[FSI] Solving solid mechanics..." << std::endl;
    if (!solid_solver_->solveDynamicStep()) {
      std::cerr << "[FSI] Solid step failed" << std::endl;
      return false;
    }

    // Step 5: Deform mesh based on solid displacement
    if (enable_mesh_update_) {
      std::cout << "[FSI] Deforming mesh..." << std::endl;
      solid_solver_->deformMesh();

      // Step 6: Rebuild fluid matrices on deformed mesh
      if (enable_matrix_rebuild_) {
        std::cout << "[FSI] Rebuilding fluid matrices..." << std::endl;
        fluid_solver_->build_sparse_matrices();
      }
    }

    // Print coupling statistics
    auto [max_disp, max_vm, max_vel] = solid_solver_->get_stats();
    std::cout << "[FSI] Coupling stats - Max displacement: " << max_disp
              << ", Max von Mises: " << max_vm << ", Max velocity: " << max_vel
              << std::endl;

    return true;
  }

  /**
   * @brief Implicit (strongly coupled) FSI step
   *
   * Algorithm:
   * Iterate until convergence:
   * 1. Solve fluid on current mesh
   * 2. Extract pressure forces
   * 3. Solve solid with pressure forces
   * 4. Update mesh and rebuild matrices
   * 5. Check convergence (displacement change)
   * 6. If not converged, repeat from step 1
   */
  bool stepImplicit() {
    std::vector<Eigen::Vector3d> disp_old =
        solid_solver_->getTotalDisplacement();

    for (size_t iter = 0; iter < max_fsi_iterations_; ++iter) {
      std::cout << "[FSI] Implicit iteration " << (iter + 1) << "/"
                << max_fsi_iterations_ << std::endl;

      // Solve fluid
      if (!fluid_solver_->pisoStep()) {
        std::cerr << "[FSI] Fluid step failed in implicit iteration" << iter
                  << std::endl;
        return false;
      }

      // Get total fluid forces (pressure + viscous shear)
      std::vector<Eigen::Vector3d> total_forces =
          fluid_solver_->computeTotalFluidForces();
      solid_solver_->setFsiTraction(total_forces);

      // Solve solid
      if (!solid_solver_->solveDynamicStep()) {
        std::cerr << "[FSI] Solid step failed in implicit iteration " << iter
                  << std::endl;
        return false;
      }

      // Check convergence
      const std::vector<Eigen::Vector3d> &disp_new =
          solid_solver_->getTotalDisplacement();
      double max_change = 0.0;
      for (size_t i = 0; i < disp_old.size(); ++i) {
        max_change =
            std::max(max_change, (disp_new[i] - disp_old[i]).norm());
      }

      std::cout << "[FSI] Displacement change: " << max_change << std::endl;

      if (max_change < fsi_tolerance_) {
        std::cout << "[FSI] Converged in " << (iter + 1) << " iterations"
                  << std::endl;

        // Update mesh with converged displacement
        if (enable_mesh_update_) {
          solid_solver_->deformMesh();
          if (enable_matrix_rebuild_) {
            fluid_solver_->build_sparse_matrices();
          }
        }
        return true;
      }

      // Update mesh for next iteration
      if (enable_mesh_update_) {
        solid_solver_->deformMesh();
        if (enable_matrix_rebuild_) {
          fluid_solver_->build_sparse_matrices();
        }
      }

      disp_old = disp_new;
    }

    std::cerr << "[FSI] Warning: Implicit coupling did not converge in "
              << max_fsi_iterations_ << " iterations" << std::endl;
    return false; // Did not converge
  }

  std::shared_ptr<NavierStokesSolver> fluid_solver_;
  std::shared_ptr<SolidMechanicsSolver> solid_solver_;
  std::shared_ptr<Mesh> mesh_ptr_;
  CouplingScheme scheme_;
  double time_;
  int step_count_;

  // Coupling parameters
  size_t max_fsi_iterations_;
  double fsi_tolerance_;
  bool enable_mesh_update_;
  bool enable_matrix_rebuild_;
};

/**
 * @brief Simple Navier-Stokes solver integrator using basic operator splitting
 *
 * Provides a lightweight time integration alternative to NavierStokesIntegrator (PISO)
 * for faster prototyping and testing. Uses single-pass momentum predictor and
 * pressure corrector without iteration.
 */
class NavierStokesSimpleSolverIntegrator {
public:
  explicit NavierStokesSimpleSolverIntegrator(std::shared_ptr<NavierStokesSimpleSolver> solver)
      : solver_(solver), time_(0.0), step_count_(0) {}

  /**
   * @brief Advance one time step using simple operator splitting
   * @return true if step succeeded
   */
  bool step() {
    if (!solver_) {
      return false;
    }

    bool success = solver_->step();
    if (success) {
      time_ = solver_->getTime();
      step_count_++;
    }
    return success;
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
    while (time_ < target_time) {
      if (!step()) {
        return false;
      }
    }
    return true;
  }

  double getTime() const { return time_; }
  int getStepCount() const { return step_count_; }
  std::shared_ptr<NavierStokesSimpleSolver> getSolver() { return solver_; }

private:
  std::shared_ptr<NavierStokesSimpleSolver> solver_;
  double time_;
  int step_count_;
};

