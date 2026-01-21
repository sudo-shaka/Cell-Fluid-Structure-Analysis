#include "DPM/DeformableParticle.hpp"
#include "DPM/ParticleInteractions.hpp"
#include "FEM/NavierStokes.hpp"
#include "FEM/SolidMechanics.hpp"
#include "Mesh/Mesh.hpp"
#include "Numerics/ThreadPool.hpp"
#include <memory>

class CoupledSolver {
private:
  std::shared_ptr<Mesh> mesh_ = nullptr;
  std::shared_ptr<NavierStokesSolver> fluid_solver_ = nullptr;
  std::shared_ptr<SolidMechanicsSolver> mechanics_solver_ = nullptr;
  std::shared_ptr<ParticleInteractions> dpm_solver_ = nullptr;
  std::unique_ptr<Fluid> fluid_properties_ = nullptr;
  std::unique_ptr<Material> material_properties_ = nullptr;

  // for dpm integration
  inline static ThreadPool pool_ = ThreadPool();
  double dpm_dt_;

public:
  // Constructor
  explicit CoupledSolver() = default;
  explicit CoupledSolver(std::shared_ptr<Mesh> mesh);

  void initializeSolidMechanicsSolver();
  void initializeNavierStokesSolver();
  void initializeDPMSolver(const std::vector<DeformableParticle> &particles);
  void updateBoundariesFromParticlePositions();
  void integrateStep();
  void mechanicsStep();
  void fluidStep();
  void dpmStep();

  // Setters
  void setMesh(const std::shared_ptr<Mesh> mesh) { mesh_ = std::move(mesh); }
  void setFluid(const Fluid &props) {
    fluid_properties_ = std::make_unique<Fluid>(std::move(props));
  }
  void setMaterial(const Material &m) {
    material_properties_ = std::make_unique<Material>(std::move(m));
  }
  void setDPMSolverDt(double dt) {
    if (dpm_solver_)
      dpm_dt_ = dt;
  }
  void setFluidSolverDt(double dt) {
    if (fluid_properties_)
      fluid_solver_->setDt(dt);
  }
  void setMechanicsDt(double dt) {
    if (mechanics_solver_)
      mechanics_solver_->setDt(dt);
  }

  // Getters
  const std::shared_ptr<ParticleInteractions> getDPMSolver() const {
    return dpm_solver_;
  }
  const std::shared_ptr<NavierStokesSolver> getFluidSolver() const {
    return fluid_solver_;
  }
  const std::shared_ptr<SolidMechanicsSolver> getMechanicsSolver() const {
    return mechanics_solver_;
  }
};
