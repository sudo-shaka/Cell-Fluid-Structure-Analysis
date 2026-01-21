
#pragma once

#include "DPM/ParticleInteractions.hpp"
#include "Mesh/Mesh.hpp"
#include "Numerics/ThreadPool.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class DPMTimeIntegrator {
public:
  explicit DPMTimeIntegrator(std::shared_ptr<ParticleInteractions> tissue_ptr)
      : tissue(std::move(tissue_ptr)) {}

  void advance_steps(int nsteps) {
    for (int i = 0; i < nsteps; i++) {
      advanceStep();
    }
  }

  inline void advanceStep() {
    (this->*integrate_)(); // call selected integrator
  }

  std::shared_ptr<ParticleInteractions> tissue;
  void setMesh(const std::shared_ptr<Mesh> m) { mesh_ = m; }
  void setDT(double newdt) { dt_ = newdt; }
  static void eulerStep(ThreadPool &pool, const std::vector<Face> &faces,
                        double dt, ParticleInteractions &particles);

private:
  double dt_ = 0.01;
  std::string integration_method_ = "euler";
  std::shared_ptr<Mesh> mesh_;

  // Pointer to member function of type void()
  using IntegratorMethod = void (DPMTimeIntegrator::*)();
  IntegratorMethod integrate_ = &DPMTimeIntegrator::eulerStep;

  inline static ThreadPool pool_ = ThreadPool();

  // Map from name to pointer-to-member
  std::unordered_map<std::string, IntegratorMethod> methods_{
      {"euler", &DPMTimeIntegrator::eulerStep},
      {"backward_euler", &DPMTimeIntegrator::backwardEulerStep},
      {"rk4", &DPMTimeIntegrator::rungKutaStep}};

  // Implementations (declarations only here)
  void saveState();
  void resetForces();
  void updateForces();
  void restoreState();

  void eulerStep();
  void backwardEulerStep();
  void rungKutaStep();
};
