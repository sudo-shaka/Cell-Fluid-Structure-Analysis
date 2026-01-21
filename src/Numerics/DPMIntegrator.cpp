#include "DPM/DeformableParticle.hpp"
#include "DPM/ParticleInteractions.hpp"
#include "Numerics/DPMIntegrator.hpp"
#include "Numerics/ThreadPool.hpp"
#include <iostream>
#include <vector>

// State Management

void DPMTimeIntegrator::saveState() {
  // Save current state for potential rollback
  // This would store positions and velocities of all particles
  // TODO: Implement state saving if needed for advanced integration schemes
}

void DPMTimeIntegrator::restoreState() {
  // Restore previously saved state
  // TODO: Implement state restoration if needed
}

void DPMTimeIntegrator::resetForces() {
  const size_t n_particles = tissue->nParticles();

  parallel_for(pool_, n_particles, [this](size_t i) -> void {
    // Access particle through tissue and reset its forces
    const_cast<DeformableParticle &>(tissue->getParticle(i)).resetForces();
  });
}

void DPMTimeIntegrator::updateForces() {
  resetForces();

  // Rebuild spatial grid for efficient neighbor queries
  const_cast<ParticleInteractions *>(tissue.get())
      ->rebuildIntercellularSpatialGrid();

  const size_t n_particles = tissue->nParticles();

  // Update shape forces (volume, area, bending) for each particle
  parallel_for(pool_, n_particles, [this](size_t i) -> void {
    const_cast<DeformableParticle &>(tissue->getParticle(i))
        .ShapeForcesUpdate();
  });

  // Update interaction forces (cell-cell, cell-matrix)
  parallel_for(pool_, n_particles, [this](size_t i) -> void {
    const_cast<ParticleInteractions *>(tissue.get())->interactingForceUpdate(i);
  });

  parallel_for(pool_, n_particles, [this](size_t i) -> void {
    const_cast<ParticleInteractions *>(tissue.get())
        ->cellMeshInteractionUpdate(this->mesh_->getFaces(), i);
  });
}

// Integration Methods
void DPMTimeIntegrator::eulerStep() {
  DPMTimeIntegrator::eulerStep(pool_, mesh_->getFaces(), dt_, *this->tissue);
}

void DPMTimeIntegrator::backwardEulerStep() {
  // Semi-implicit Euler (backward Euler):
  // Calculate forces at current positions (F_n)
  // Make tentative step with forward Euler
  // Calculate forces at new positions (F_{n+1})
  // Average forces and update with averaged forces

  const size_t n_particles = tissue->nParticles();

  // Step 1: Calculate forces at current positions
  updateForces();

  // Step 2: Store current positions
  std::vector<std::vector<Eigen::Vector3d>> old_positions(n_particles);
  for (size_t i = 0; i < n_particles; ++i) {
    const auto &particle = tissue->getParticle(i);
    const auto &geom = particle.getGeometry();
    old_positions[i].resize(geom.nVerts());
    for (size_t j = 0; j < geom.nVerts(); ++j) {
      old_positions[i][j] = geom.getPosition(j);
    }
  }

  // Step 3: Make tentative forward Euler step
  parallel_for(pool_, n_particles, [this](size_t i) -> void {
    const_cast<DeformableParticle &>(tissue->getParticle(i))
        .eulerUpdatePositions(dt_);
  });

  // Step 4: Store tentative forces
  std::vector<std::vector<Eigen::Vector3d>> tentative_forces(n_particles);
  for (size_t i = 0; i < n_particles; ++i) {
    const auto &particle = tissue->getParticle(i);
    tentative_forces[i] = particle.getTotalForces();
  }

  // Step 5: Calculate forces at tentative new positions
  updateForces();

  // Step 6: Average forces and move from original positions with averaged
  // forces
  parallel_for(
      pool_, n_particles,
      [this, &old_positions, &tentative_forces](size_t i) -> void {
        auto &particle =
            const_cast<DeformableParticle &>(tissue->getParticle(i));

        // Get current (new) forces
        const auto &new_forces = particle.getTotalForces();

        // Restore old positions
        auto &shape = const_cast<Polyhedron &>(particle.getGeometry());
        for (size_t j = 0; j < old_positions[i].size(); ++j) {
          shape.getMutPosition(j) = old_positions[i][j];
        }

        // Average the forces: F_avg = (F_old + F_new) / 2
        std::vector<Eigen::Vector3d> averaged_forces(new_forces.size());
        for (size_t j = 0; j < new_forces.size(); ++j) {
          averaged_forces[j] = 0.5 * (tentative_forces[i][j] + new_forces[j]);
        }

        // Apply averaged forces manually
        // x_{n+1} = x_n + dt * F_avg
        for (size_t j = 0; j < old_positions[i].size(); ++j) {
          shape.getMutPosition(j) += dt_ * averaged_forces[j];
        }

        // Update geometry after position change
        const_cast<Polyhedron &>(shape).updateGeometry();
      });
}

void DPMTimeIntegrator::rungKutaStep() {
  // 4th order Runge-Kutta integration
  // k1 = F(x_n)
  // k2 = F(x_n + dt/2 * k1)
  // k3 = F(x_n + dt/2 * k2)
  // k4 = F(x_n + dt * k3)
  // x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

  const size_t n_particles = tissue->nParticles();

  // Save initial positions
  std::vector<std::vector<Eigen::Vector3d>> x0(n_particles);
  for (size_t i = 0; i < n_particles; ++i) {
    const auto &geom = tissue->getParticle(i).getGeometry();
    x0[i].resize(geom.nVerts());
    for (size_t j = 0; j < geom.nVerts(); ++j) {
      x0[i][j] = geom.getPosition(j);
    }
  }

  // k1: Evaluate forces at current position
  updateForces();
  std::vector<std::vector<Eigen::Vector3d>> k1(n_particles);
  for (size_t i = 0; i < n_particles; ++i) {
    k1[i] = tissue->getParticle(i).getTotalForces();
  }

  // Move to x_n + dt/2 * k1
  parallel_for(pool_, n_particles, [this, &x0, &k1](size_t i) -> void {
    auto &particle = const_cast<DeformableParticle &>(tissue->getParticle(i));
    auto &geom = const_cast<Polyhedron &>(particle.getGeometry());
    for (size_t j = 0; j < x0[i].size(); ++j) {
      geom.getMutPosition(j) = x0[i][j] + 0.5 * dt_ * k1[i][j];
    }
    geom.updateGeometry();
  });

  // k2: Evaluate forces at x_n + dt/2 * k1
  updateForces();
  std::vector<std::vector<Eigen::Vector3d>> k2(n_particles);
  for (size_t i = 0; i < n_particles; ++i) {
    k2[i] = tissue->getParticle(i).getTotalForces();
  }

  // Move to x_n + dt/2 * k2
  parallel_for(pool_, n_particles, [this, &x0, &k2](size_t i) -> void {
    auto &particle = const_cast<DeformableParticle &>(tissue->getParticle(i));
    auto &geom = const_cast<Polyhedron &>(particle.getGeometry());
    for (size_t j = 0; j < x0[i].size(); ++j) {
      geom.getMutPosition(j) = x0[i][j] + 0.5 * dt_ * k2[i][j];
    }
    geom.updateGeometry();
  });

  // k3: Evaluate forces at x_n + dt/2 * k2
  updateForces();
  std::vector<std::vector<Eigen::Vector3d>> k3(n_particles);
  for (size_t i = 0; i < n_particles; ++i) {
    k3[i] = tissue->getParticle(i).getTotalForces();
  }

  // Move to x_n + dt * k3
  parallel_for(pool_, n_particles, [this, &x0, &k3](size_t i) -> void {
    auto &particle = const_cast<DeformableParticle &>(tissue->getParticle(i));
    auto &geom = const_cast<Polyhedron &>(particle.getGeometry());
    for (size_t j = 0; j < x0[i].size(); ++j) {
      geom.getMutPosition(j) = x0[i][j] + dt_ * k3[i][j];
    }
    geom.updateGeometry();
  });

  // k4: Evaluate forces at x_n + dt * k3
  updateForces();
  std::vector<std::vector<Eigen::Vector3d>> k4(n_particles);
  for (size_t i = 0; i < n_particles; ++i) {
    k4[i] = tissue->getParticle(i).getTotalForces();
  }

  // Final update: x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
  parallel_for(
      pool_, n_particles, [this, &x0, &k1, &k2, &k3, &k4](size_t i) -> void {
        auto &particle =
            const_cast<DeformableParticle &>(tissue->getParticle(i));
        auto &geom = const_cast<Polyhedron &>(particle.getGeometry());

        for (size_t j = 0; j < x0[i].size(); ++j) {
          geom.getMutPosition(j) = x0[i][j] + dt_ / 6.0 *
                                                  (k1[i][j] + 2.0 * k2[i][j] +
                                                   2.0 * k3[i][j] + k4[i][j]);
        }
        geom.updateGeometry();
      });

  std::cout << "[DPMIntegrator] RK4 step completed" << std::endl;
}

void DPMTimeIntegrator::eulerStep(ThreadPool &pool,
                                  const std::vector<Face> &faces, double dt,
                                  ParticleInteractions &particles) {
  // Forward Euler integration: x_{n+1} = x_n + dt * v_n
  // where v_n = F_n / m (assuming unit mass)
  const size_t n_cells = particles.nParticles();
  parallel_for(pool, n_cells, [&particles](size_t i) -> void {
    // Access particle through tissue and reset its forces
    const_cast<DeformableParticle &>(particles.getParticle(i)).resetForces();
  });
  // Rebuild spatial grid for efficient neighbor queries
  particles.rebuildIntercellularSpatialGrid();

  // Update shape forces (volume, area, bending) for each particle
  parallel_for(pool, n_cells, [&particles](size_t i) -> void {
    const_cast<DeformableParticle &>(particles.getParticle(i))
        .ShapeForcesUpdate();
  });

  // Update interaction forces (cell-cell, cell-matrix)
  parallel_for(pool, n_cells, [&particles](size_t i) -> void {
    particles.interactingForceUpdate(i);
  });

  parallel_for(pool, n_cells, [&particles, &faces](size_t i) -> void {
    particles.cellMeshInteractionUpdate(faces, i);
  });

  // Euler update
  parallel_for(pool, n_cells, [&particles, &dt](size_t i) -> void {
    const_cast<DeformableParticle &>(particles.getParticle(i))
        .eulerUpdatePositions(dt);
  });
}
