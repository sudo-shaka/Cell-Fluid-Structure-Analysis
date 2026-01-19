#include "FEM/NavierStokes.hpp"
#include "FEM/SolidMechanics.hpp"
#include "IO/VtkExport.hpp"
#include "Mesh/Mesh.hpp"
#include "Numerics/FEMIntegrator.hpp"
#include <iostream>
#include <memory>

/**
 * @brief Example demonstrating Fluid-Structure Interaction (FSI) coupling
 *
 * This example simulates a flexible tube/channel with fluid flow.
 * The fluid (Navier-Stokes) applies pressure forces on the structure,
 * which deforms (Solid Mechanics), and the deformation affects the fluid flow.
 *
 * Setup:
 * - Rectangular channel mesh
 * - Fluid inlet at one end, outlet at the other
 * - Walls are fixed for structure, no-slip for fluid
 * - Two-way coupling: pressure forces -> displacement -> mesh deformation
 */
int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "  FSI Coupling Example - Flexible Tube" << std::endl;
  std::cout << "========================================\n" << std::endl;

  // ============================================================
  // 1. Create shared mesh (rectangular channel)
  // ============================================================
  std::cout << "[1] Creating mesh..." << std::endl;

  double length = 1.0;  // Channel length (flow direction)
  double width = 0.2;   // Channel width
  double height = 0.2;  // Channel height
  int nx = 40;          // Elements in x (flow direction)
  int ny = 10;           // Elements in y
  int nz = 10;           // Elements in z

  auto mesh_ptr = std::make_shared<Mesh>(
      Mesh::structuredRectangularPrism(length, width, height, nx, ny, nz));

  std::cout << "  Mesh created: " << mesh_ptr->nVertices() << " vertices, "
            << mesh_ptr->nTets() << " tets, " << mesh_ptr->nFaces()
            << " faces\n"
            << std::endl;

  // ============================================================
  // 2. Set up boundary conditions
  // ============================================================
  std::cout << "[2] Setting boundary conditions..." << std::endl;

  // Define inlet/outlet direction (flow in +x direction)
  Eigen::Vector3d flow_direction(1.0, 0.0, 0.0);
  double inlet_outlet_coverage = 0.1; // 10% at each end

  // Set fluid BCs (inlet, outlet, walls)
  Mesh::setupBoundaryConditions(flow_direction, inlet_outlet_coverage,
                                 *mesh_ptr);

  // Set solid BCs (fix inlet and outlet ends)
  const auto &vertices = mesh_ptr->getVertPositions();
  for (size_t i = 0; i < vertices.size(); ++i) {
    const Eigen::Vector3d &pos = vertices[i];

    // Fix vertices at inlet (x near 0) and outlet (x near length)
    if (pos.x() < 0.05 * length || pos.x() > 0.95 * length) {
      mesh_ptr->setSolidVertexBC(i, SolidBCType::Fixed);
    } else {
      mesh_ptr->setSolidVertexBC(i, SolidBCType::Free);
    }
  }

  // Set P2 fluid boundary conditions from P1
  mesh_ptr->setP2BoundariesFromP1Boundaries();

  std::cout << "  Boundary conditions set\n" << std::endl;

  // ============================================================
  // 3. Create and initialize Navier-Stokes solver
  // ============================================================
  std::cout << "[3] Initializing Navier-Stokes solver..." << std::endl;

  auto fluid_solver = std::make_shared<NavierStokesSolver>();

  // Fluid properties (blood-like)
  Fluid fluid_props;
  fluid_props.density = 1060.0;     // kg/m^3
  fluid_props.viscosity = 0.004;    // Pa·s
  fluid_props.turbuelence_model = TurbulenceModel::Laminar;
  fluid_props.viscosity_model = ViscosityModel::Newtonian;

  fluid_solver->initialize(mesh_ptr, fluid_props);

  // Set inlet velocity
  Eigen::Vector3d inlet_velocity(0.1, 0.0, 0.0); // 0.1 m/s in x-direction
  fluid_solver->setMeanInletVelocity(inlet_velocity);

  // Set outlet pressure
  fluid_solver->setOutletType(OutletType::DirichletPressure);
  fluid_solver->setOutletPressure(0.0);

  // Set time step
  double dt = 1e-3; // 1 ms
  fluid_solver->setDt(dt);

  std::cout << "  Fluid solver initialized\n" << std::endl;

  // ============================================================
  // 4. Create and initialize Solid Mechanics solver
  // ============================================================
  std::cout << "[4] Initializing Solid Mechanics solver..." << std::endl;

  auto solid_solver = std::make_shared<SolidMechanicsSolver>();

  // Material properties (soft tissue / rubber-like)
  Material material = Material::linear_elastic(
      1e6,   // Young's modulus: 1 MPa (soft)
      0.45,  // Poisson's ratio (nearly incompressible)
      1100.0 // Density: kg/m^3
  );

  // Add damping for stability
  material = material.with_damping(0.01, 0.001);

  solid_solver->initialize(mesh_ptr, material);

  // Set time step (same as fluid)
  solid_solver->setDt(dt);

  // Use Newmark-beta parameters for stability
  solid_solver->setNewmarkParams(0.25, 0.5); // Average acceleration method

  std::cout << "  Solid solver initialized\n" << std::endl;

  // ============================================================
  // 5. Create FSI coupled integrator
  // ============================================================
  std::cout << "[5] Creating FSI coupled integrator..." << std::endl;

  // Choose coupling scheme:
  // - Explicit: faster, less stable, suitable for weak coupling
  // - Implicit: slower, more stable, suitable for strong coupling
  auto fsi_integrator = std::make_shared<FSICoupledIntegrator>(
      fluid_solver, solid_solver,
      FSICoupledIntegrator::CouplingScheme::Explicit);

  // Set FSI coupling parameters
  fsi_integrator->setMaxFSIIterations(10);
  fsi_integrator->setFSITolerance(1e-4);
  fsi_integrator->enableMeshUpdate(true);
  fsi_integrator->enableMatrixRebuild(true);

  std::cout << "  FSI integrator created\n" << std::endl;

  // ============================================================
  // 6. Run coupled simulation
  // ============================================================
  std::cout << "[6] Running coupled FSI simulation..." << std::endl;

  int num_steps = 100;
  int output_frequency = 10;

  for (int step = 0; step < num_steps; ++step) {
    std::cout << "\n========== Time Step " << (step + 1) << " / " << num_steps
              << " ==========" << std::endl;
    std::cout << "Time: " << fsi_integrator->getTime() << " s" << std::endl;

    // Advance one coupled time step
    if (!fsi_integrator->step()) {
      std::cerr << "ERROR: FSI simulation failed at step " << (step + 1)
                << std::endl;
      return 1;
    }

    // Output results periodically
    if (step % output_frequency == 0 || step == num_steps - 1) {
      std::string filename =
          "fsi_output_" + std::to_string(step / output_frequency);

      // Export fluid results
      VtkExport::exportMeshWithVelocityAndPressure(
          *mesh_ptr, *fluid_solver, filename + "_fluid.vtk");

      // Export solid results
      VtkExport::exportMeshWithDisplacementAndStress(
          *mesh_ptr, *solid_solver, filename + "_solid.vtk");

      std::cout << "  Output written: " << filename << "_*.vtk" << std::endl;
    }

    // Print statistics
    auto [max_disp, max_vm, max_vel_solid] = solid_solver->get_stats();
    const auto &velocity = fluid_solver->getVelocity();
    double max_vel_fluid = 0.0;
    for (const auto &v : velocity) {
      max_vel_fluid = std::max(max_vel_fluid, v.norm());
    }

    std::cout << "  Max fluid velocity: " << max_vel_fluid << " m/s"
              << std::endl;
    std::cout << "  Max solid displacement: " << max_disp << " m" << std::endl;
    std::cout << "  Max von Mises stress: " << max_vm << " Pa" << std::endl;
  }

  std::cout << "\n========================================" << std::endl;
  std::cout << "  FSI simulation completed successfully!" << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}
