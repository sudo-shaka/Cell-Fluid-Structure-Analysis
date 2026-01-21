#include "Numerics/CoupledSolver.hpp"
#include "Numerics/DPMIntegrator.hpp"

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
  dpm_solver_ = std::make_shared<ParticleInteractions>(particles);
}

void CoupledSolver::dpmStep() {
  if (!dpm_solver_)
    return;
  DPMTimeIntegrator::eulerStep(pool_, mesh_->getFaces(), dpm_dt_, *dpm_solver_);
}

void CoupledSolver::fluidStep() {
  if (!fluid_solver_)
    return;
  fluid_solver_->pisoStep();
}

void CoupledSolver::mechanicsStep() {
  if (!mechanics_solver_)
    return;
  mechanics_solver_->solveDynamicStep();
  mechanics_solver_->deformMesh();
}

void CoupledSolver::integrateStep() {
  fluidStep();
  mechanicsStep();
  dpmStep();

  // TODO solver adaptations to new mesh
}

void CoupledSolver::updateBoundariesFromParticlePositions() {
  if (!mesh_ || !dpm_solver_) {
    return;
  }

  // Update P1 nodes (primary vertices)
  const size_t n_vertices = mesh_->nVertices();
  std::vector<char> p1_inside(n_vertices, 0);

  parallel_for(pool_, n_vertices, [&](size_t vi) {
    const FluidBCType fbc = mesh_->getFluidVertexBC(vi);
    const SolidBCType sbc = mesh_->getSolidVertexBC(vi);

    // Preserve inlet/outlet/fixed constraints
    if (fbc == FluidBCType::Inlet || fbc == FluidBCType::Outlet ||
        sbc == SolidBCType::Fixed) {
      return;
    }

    for (size_t pi = 0; pi < dpm_solver_->nParticles(); ++pi) {
      if (Polyhedron::pointInside(dpm_solver_->getParticle(pi).getGeometry(),
                                  mesh_->getVertexPositon(vi))) {
        p1_inside[vi] = 1;
        break;
      }
    }
  });

  for (size_t vi = 0; vi < n_vertices; ++vi) {
    if (p1_inside[vi]) {
      mesh_->setFluidVertexBC(vi, FluidBCType::Wall); // no-slip
      mesh_->setSolidVertexBC(vi, SolidBCType::Free); // free solid motion
    }
  }

  // Build edge -> P1 mapping for P2 nodes
  const size_t n_edge_nodes = mesh_->getNumberOfEdgeNodes();
  std::vector<std::pair<int, int>> edge_endpoints(n_edge_nodes,
                                                  std::make_pair(-1, -1));

  const auto &tets = mesh_->getTets();
  for (size_t ti = 0; ti < tets.size(); ++ti) {
    const auto &edge_nodes = mesh_->getTetEdgeNodes(ti);
    const auto &verts = tets[ti].vertids;

    const int edge_pairs[6][2] = {{0, 1}, {0, 2}, {0, 3},
                                  {1, 2}, {1, 3}, {2, 3}};

    for (int e = 0; e < 6; ++e) {
      const int p2_idx = edge_nodes[e];
      if (p2_idx < 0)
        continue;

      const int v1 = verts[edge_pairs[e][0]];
      const int v2 = verts[edge_pairs[e][1]];

      if (edge_endpoints[p2_idx].first == -1) {
        edge_endpoints[p2_idx] = std::make_pair(v1, v2);
      }
    }
  }

  // Update P2 nodes (edge midpoints)
  std::vector<char> p2_inside(edge_endpoints.size(), 0);

  parallel_for(pool_, edge_endpoints.size(), [&](size_t p2_idx) {
    const auto [v1, v2] = edge_endpoints[p2_idx];
    if (v1 < 0 || v2 < 0)
      return; // should not happen, but guard just in case

    const FluidBCType bc1 = mesh_->getFluidVertexBC(v1);
    const FluidBCType bc2 = mesh_->getFluidVertexBC(v2);
    const SolidBCType sbc1 = mesh_->getSolidVertexBC(v1);
    const SolidBCType sbc2 = mesh_->getSolidVertexBC(v2);

    // Skip edges touching inlet/outlet/fixed nodes
    if (bc1 == FluidBCType::Inlet || bc1 == FluidBCType::Outlet ||
        bc2 == FluidBCType::Inlet || bc2 == FluidBCType::Outlet ||
        sbc1 == SolidBCType::Fixed || sbc2 == SolidBCType::Fixed) {
      return;
    }

    const Eigen::Vector3d midpoint =
        0.5 * (mesh_->getVertexPositon(v1) + mesh_->getVertexPositon(v2));

    for (size_t pi = 0; pi < dpm_solver_->nParticles(); ++pi) {
      if (Polyhedron::pointInside(dpm_solver_->getParticle(pi).getGeometry(),
                                  midpoint)) {
        p2_inside[p2_idx] = 1;
        break;
      }
    }
  });

  for (size_t p2_idx = 0; p2_idx < edge_endpoints.size(); ++p2_idx) {
    if (p2_inside[p2_idx]) {
      mesh_->setFluidP2vertexBC(p2_idx, FluidBCType::Wall); // no-slip
    }
  }
}
