#include "FEM/SolidMechanics.hpp"
#include "IO/VtkExport.hpp"
#include "Mesh/Mesh.hpp"
#include "Numerics/FEMIntegrator.hpp"
#include <iostream>
#include <memory>

int main() {
  std::cout << "=== Structural Dynamics Simulation ===" << std::endl;

  // Geometry parameters (cantilever beam)
  const double length = 1.0;
  const double width = 0.1;
  const double height = 0.1;
  const int nx = 20;
  const int ny = 3;
  const int nz = 3;

  // Material properties (steel-like)
  const double youngs_modulus = 200e9; // Pa
  const double poissons_ratio = 0.3;
  const double density = 7850.0; // kg/m³

  // Time integration parameters
  const double dt = 1e-6;          // Time step size (smaller for stability)
  const double total_time = 0.001; // Total simulation time
  const int output_interval = 100; // Output every N steps

  std::cout << "\n[Setup] Creating cantilever beam mesh..." << std::endl;
  auto mesh = std::make_shared<Mesh>(
      Mesh::structuredRectangularPrism(length, width, height, nx, ny, nz));

  std::cout << "[Setup] Mesh created with " << mesh->nVertices()
            << " vertices, " << mesh->nTets() << " tetrahedra" << std::endl;

  // Setup boundary conditions - fix left end (x = 0)
  for (size_t i = 0; i < mesh->nVertices(); ++i) {
    const auto &pos = mesh->getVertexPositon(i);
    if (pos.x() < 1e-6) { // Left end
      mesh->setSolidVertexBC(i, SolidBCType::Fixed);
    } else {
      mesh->setSolidVertexBC(i, SolidBCType::Free);
    }
  }

  // Create material
  Material steel =
      Material::linear_elastic(youngs_modulus, poissons_ratio, density);
  steel =
      steel.with_damping(0.1, 0.0001); // Add significant damping for stability

  std::cout << "[Setup] Initializing solid mechanics solver..." << std::endl;
  auto solid_solver = std::make_shared<SolidMechanicsSolver>();
  solid_solver->initialize(mesh, steel);
  solid_solver->setDt(dt);
  solid_solver->setNewmarkParams(0.25, 0.5); // Average acceleration method

  // Apply gravity load
  Eigen::Vector3d gravity(0.0, 0.0, -9.81);
  solid_solver->applyGravity(gravity);

  std::cout << "[Setup] Material properties:" << std::endl;
  std::cout << "  Young's modulus: " << youngs_modulus / 1e9 << " GPa"
            << std::endl;
  std::cout << "  Poisson's ratio: " << poissons_ratio << std::endl;
  std::cout << "  Density: " << density << " kg/m³" << std::endl;

  // Create integrator for dynamic analysis
  SolidMechanicsIntegrator integrator(
      solid_solver, SolidMechanicsIntegrator::IntegrationType::Dynamic);

  std::cout << "\n[Simulation] Starting dynamic analysis..." << std::endl;
  std::cout << "  Time step: " << dt << " s" << std::endl;
  std::cout << "  Total time: " << total_time << " s" << std::endl;
  std::cout << "  Expected steps: " << static_cast<int>(total_time / dt)
            << std::endl;

  int step = 0;

  while (integrator.getTime() < total_time) {
    // Perform one time step
    bool success = integrator.step();

    if (!success) {
      std::cerr << "[Error] Simulation failed at step " << step
                << ", time = " << integrator.getTime() << std::endl;
      return 1;
    }

    // Periodic output
    if (step % output_interval == 0) {
      // Get statistics
      auto [max_disp, max_vm, max_vel] = solid_solver->get_stats();

      std::cout << "  Step " << step << ": t = " << integrator.getTime() << " s"
                << std::endl;
      std::cout << "    Max displacement: " << max_disp << " m" << std::endl;
      std::cout << "    Max von Mises: " << max_vm / 1e6 << " MPa" << std::endl;
      std::cout << "    Max velocity: " << max_vel << " m/s" << std::endl;

      // Export VTK for visualization
      if (step % (output_interval * 5) == 0) {
        std::string filename = "structural_dynamics_t" +
                               std::to_string(integrator.getTime()) + ".vtk";
        io::exportToVtk(filename, *solid_solver);
        std::cout << "    Output: " << filename << std::endl;
      }
    }

    step++;
  }

  std::cout << "\n[Complete] Simulation finished successfully!" << std::endl;
  std::cout << "  Total steps: " << integrator.getStepCount() << std::endl;
  std::cout << "  Final time: " << integrator.getTime() << " s" << std::endl;

  // Final statistics
  auto [max_disp, max_vm, max_vel] = solid_solver->get_stats();
  std::cout << "\n[Final State]" << std::endl;
  std::cout << "  Maximum displacement: " << max_disp << " m" << std::endl;
  std::cout << "  Maximum von Mises stress: " << max_vm / 1e6 << " MPa"
            << std::endl;
  std::cout << "  Maximum velocity: " << max_vel << " m/s" << std::endl;

  return 0;
}
