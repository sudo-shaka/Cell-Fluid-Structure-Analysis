#include "FEM/NavierStokes.hpp"
#include "Numerics/CoupledSolver.hpp"
#include "Numerics/DPMIntegrator.hpp"
#include <stdexcept>

CoupledSolver::CoupledSolver(const std::shared_ptr<Mesh> mesh) {
  if (!mesh) {
    std::cerr << "[Coupled Solver] trying to construct with mesh_ptr == null\n";
    std::cerr << "[Coupled Solver] Cannot construct solver without valid mesh."
              << std::endl;
    return;
  }
  mesh_ = std::move(mesh);
  const auto &bc = mesh_->getFluidVertexBC(0);
  if (bc == FluidBCType::Undefined) {
    std::cerr << "[Coupled Solver] No boundaries appled to mesh. Solvers will "
                 "not update"
              << std::endl;
  }
  original_fluid_bc_.resize(mesh_->nVertices());
  original_solid_bc_.resize(mesh_->nVertices());
  for (size_t vi = 0; vi < mesh_->nVertices(); ++vi) {
    original_fluid_bc_[vi] = mesh_->getFluidVertexBC(vi);
    original_solid_bc_[vi] = mesh_->getSolidVertexBC(vi);
  }
}

void CoupledSolver::initializeNavierStokesSolver() {
  if (!mesh_) {
    std::cerr << "[Coupled Solver] cannot initialize when mesh is NULL\n";
    return;
  }
  if (!fluid_properties_) {
    std::cerr << "[coupled solver]. fluild fluid properties are not set. "
                 "please set first before trying to initialize solver"
              << std::endl;
    return;
  }
  // Create solver instance if it doesn't exist
  if (!fluid_solver_) {
    fluid_solver_ = std::make_shared<NavierStokesSolver>();
  }
  fluid_solver_->initialize(mesh_, *fluid_properties_);
  // may not be required but just initialize the vector
  fluid_forces_ = fluid_solver_->computeTotalFluidForces();
  // right now only Dirchlet outlet pressure boundaries allowed.
  fluid_solver_->setOutletType(OutletType::DirichletPressure);
}

void CoupledSolver::initializeSolidMechanicsSolver() {
  if (!mesh_) {
    std::cerr << "[Coupled Solver] cannot initialize when mesh is NULL\n";
    return;
  }
  if (!material_properties_) {
    std::cerr << "[Coupled Solver] material properties not set. Set properties "
                 "before intiaization"
              << std::endl;
    return;
  }
  // Create solver instance if it doesn't exist
  if (!mechanics_solver_) {
    mechanics_solver_ = std::make_shared<SolidMechanicsSolver>();
  }
  mechanics_solver_->initialize(mesh_, *material_properties_);
}

void CoupledSolver::initializeDPMSolver(
    const std::vector<DeformableParticle> &particles) {
  if (!mesh_) {
    std::cout << "[Coupled Solver] Mesh is required for DPM initialization\n";
    return;
  }
  dpm_solver_ = std::make_shared<ParticleInteractions>(particles);
  dpm_solver_->disperseCellsToFaceCenters(mesh_->getFaces());
}

void CoupledSolver::dpmStep() {
  if (!dpm_solver_)
    return;
  DPMTimeIntegrator::eulerStep(pool_, mesh_->getFaces(), dpm_dt_, *dpm_solver_);
  std::cout << "[Coupled Solver] DPM Step complete." << std::endl;
  dpm_solver_->removeDegenerateParticles();
}

void CoupledSolver::fluidStep() {
  if (!fluid_solver_)
    return;
  bool solved = fluid_solver_->pisoStep();
  if (!solved) {
    throw std::runtime_error(
        "[Coupled Solver] Navier Stokes PISO step failed\n");
  }
  if (mechanics_solver_)
    fluid_forces_ = fluid_solver_->computeTotalFluidForces();
  std::cout << "[Coupled Solver] Fluid Step complete." << std::endl;
}

void CoupledSolver::mechanicsStep() {
  if (!mechanics_solver_)
    return;
  if (fluid_solver_ && fluid_forces_.size() > 0)
    mechanics_solver_->setFsiTraction(fluid_forces_);
  bool solved = mechanics_solver_->solveDynamicStep();
  if (!solved) {
    throw std::runtime_error(
        "[Coupled Solver] Mechanics dynamic step failed\n");
  }
  mechanics_solver_->deformMesh();
  std::cout << "[Coupled Solver] Mechanics Step complete." << std::endl;
}

void CoupledSolver::integrateStep() {
  updateBoundariesFromParticlePositions();
  // Rebuild matrices after boundary changes
  if (fluid_solver_ && (mechanics_solver_ || dpm_solver_)) {
    fluid_solver_->build_sparse_matrices();
  }
  if (mechanics_solver_ && dpm_solver_) {
    mechanics_solver_->rebuildSparseMatrices();
  }
  if (fluid_solver_ && dpm_solver_) {
    interpolateFluidForcesToParticles();
  }
  // solve with updated boundaries
  fluidStep();
  mechanicsStep();
  dpmStep();
}

void CoupledSolver::restoreOriginalBoundaries() {
  if (!mesh_) {
    return;
  }
  for (size_t vi = 0; vi < mesh_->nVertices(); ++vi) {
    mesh_->setFluidVertexBC(vi, original_fluid_bc_[vi]);
    mesh_->setSolidVertexBC(vi, original_solid_bc_[vi]);
  }
}

void CoupledSolver::updateBoundariesFromParticlePositions() {
  if (!mesh_ || !dpm_solver_) {
    return;
  }

  // Restore boundaries to original state first
  restoreOriginalBoundaries();

  // Update P1 nodes (primary vertices)
  const size_t n_vertices = mesh_->nVertices();
  std::vector<bool> set_wall(n_vertices, false);

  parallel_for(pool_, n_vertices, [&](size_t vi) {
    const FluidBCType fbc = mesh_->getFluidVertexBC(vi);
    // Preserve inlet/outlet/fixed constraints
    if (fbc != FluidBCType::Internal) {
      return;
    }
    for (size_t pi = 0; pi < dpm_solver_->nParticles(); ++pi) {
      if (Polyhedron::pointInside(dpm_solver_->getParticle(pi).getGeometry(),
                                  mesh_->getVertexPositon(vi))) {
        set_wall[vi] = true;
        break;
      }
    }
  });

  for (size_t vi = 0; vi < n_vertices; ++vi) {
    if (set_wall[vi]) {
      mesh_->setFluidVertexBC(vi, FluidBCType::Wall); // no-slip
      mesh_->setSolidVertexBC(vi, SolidBCType::Free); // free solid motion
    }
  }
  mesh_->setP2BoundariesFromP1Boundaries();
}

void CoupledSolver::interpolateFluidForcesToParticles() {
  if (!mesh_ || !dpm_solver_ || !fluid_solver_) {
    return;
  }

  // TODO: This really isn't the best way to do this since we calculated
  // interactions already + this is O(N^2)

  // Get pressure and shear forces from fluid solver
  std::vector<Eigen::Vector3d> pressure_forces =
      fluid_solver_->computePressureForces();
  std::vector<Eigen::Vector3d> shear_forces =
      fluid_solver_->computeShearStress();

  const size_t n_particles = dpm_solver_->nParticles();

  // For each particle, interpolate forces onto its vertices
  parallel_for(pool_, n_particles, [&](size_t pi) {
    auto &particle = dpm_solver_->getMutParticle(pi);
    const auto &geometry = particle.getGeometry();
    const size_t n_verts = geometry.nVerts();

    // For each vertex of the particle
    for (size_t vi = 0; vi < n_verts; ++vi) {
      const Eigen::Vector3d &particle_vert_pos = geometry.getPosition(vi);

      Eigen::Vector3d interpolated_pressure = Eigen::Vector3d::Zero();
      Eigen::Vector3d interpolated_shear = Eigen::Vector3d::Zero();
      double total_weight = 0.0;

      // Find neighboring internal mesh vertices and interpolate using inverse
      // distance weighting
      const size_t n_mesh_verts = mesh_->nVertices();
      double max_search_radius = particle.getR0(); // Adjust based on mesh size
      double epsilon = 1e-10;                      // To avoid division by zero

      for (size_t mi = 0; mi < n_mesh_verts; ++mi) {
        const FluidBCType fbc = mesh_->getFluidVertexBC(mi);

        // Only interpolate from internal vertices (where forces are computed)
        if (fbc != FluidBCType::Internal) {
          continue;
        }

        const Eigen::Vector3d &mesh_vert_pos = mesh_->getVertexPositon(mi);
        const double distance = (particle_vert_pos - mesh_vert_pos).norm();

        // Use only nearby vertices for interpolation
        if (distance > max_search_radius) {
          continue;
        }

        // Inverse distance weighting: w = 1 / (d + epsilon)
        const double weight = 1.0 / (distance + epsilon);

        interpolated_pressure += weight * pressure_forces[mi];
        interpolated_shear += weight * shear_forces[mi];
        total_weight += weight;
      }

      // Normalize by total weight
      if (total_weight > epsilon) {
        interpolated_pressure /= total_weight;
        interpolated_shear /= total_weight;
      }

      // Set the interpolated forces on the particle vertex
      particle.setPressureForce(vi, interpolated_pressure);
      particle.setShearForce(vi, interpolated_shear);
    }
  });
}
