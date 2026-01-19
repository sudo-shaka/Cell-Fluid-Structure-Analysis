#pragma once

#include "FEM/NavierStokes.hpp"
#include "FEM/SolidMechanics.hpp"
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
