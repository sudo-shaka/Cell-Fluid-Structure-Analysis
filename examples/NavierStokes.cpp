#include "FEM/NavierStokes.hpp"
#include "IO/VtkExport.hpp"
#include "Mesh/Mesh.hpp"
#include "Numerics/FEMIntegrator.hpp"
#include <iostream>
#include <memory>
#include <string>

int main() {
  std::cout << "=== Navier-Stokes Flow Simulation ===" << std::endl;

  // Flow parameters
  const double inlet_velocity = 0.1;  // m/s
  const double outlet_pressure = 0.0; // Pa (gauge)

  // Fluid properties (water at 20°C)
  const double density = 1000.0;  // kg/m³
  const double viscosity = 0.001; // Pa·s

  // Time integration parameters
  const double dt = 0.0001;       // Time step size
  const double total_time = 0.01; // Total simulation time
  const int output_interval = 10; // Output every N steps

  double radius = 2.0;
  double length = 20.0;
  int n_surface_points = 40;
  double max_edge_length = 0.2;

  std::cout << "\n[Setup] Creating rectangular channel mesh..." << std::endl;
  auto mesh = std::make_shared<Mesh>(Mesh::fromPolyhedron(
      Polyhedron::cylendar(length, radius, n_surface_points), max_edge_length));

  std::cout << "[Setup] Mesh created with " << mesh->nVertices()
            << " vertices, " << mesh->nTets() << " tetrahedra" << std::endl;

  // Setup boundary conditions
  Eigen::Vector3d flow_direction(1, 0, 0);
  Mesh::setupBoundaryConditions(flow_direction, 0.5, *mesh);

  std::cout << "[Setup] Initializing Navier-Stokes solver..." << std::endl;
  auto ns_solver = std::make_shared<NavierStokesSolver>();
  ns_solver->setMeanInletVelocity(Eigen::Vector3d(inlet_velocity, 0.0, 0.0));

  Fluid water;
  water.density = density;
  water.viscosity = viscosity;

  ns_solver->initialize(mesh, water);
  ns_solver->setDt(dt);
  ns_solver->setOutletType(OutletType::DirichletPressure);
  ns_solver->setOutletPressure(outlet_pressure);

  std::cout << "[Setup] Fluid properties:" << std::endl;
  std::cout << "  Density: " << density << " kg/m³" << std::endl;
  std::cout << "  Viscosity: " << viscosity << " Pa·s" << std::endl;
  std::cout << "  Inlet velocity: " << inlet_velocity << " m/s" << std::endl;
  std::cout << "  Reynolds number: "
            << (density * inlet_velocity * max_edge_length / viscosity)
            << std::endl;

  // Create integrator
  NavierStokesIntegrator integrator(ns_solver);

  std::cout << "\n[Simulation] Starting transient flow analysis..."
            << std::endl;
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
      // Get max velocity
      double max_vel = 0.0;
      const auto &velocities = ns_solver->getVelocity();
      for (const auto &v : velocities) {
        max_vel = std::max(max_vel, v.norm());
      }

      std::cout << "  Step " << step << ": t = " << integrator.getTime()
                << " s, max velocity = " << max_vel << " m/s" << std::endl;

      // Export VTK for visualization
      if (step % (output_interval * 5) == 0) {
        std::string filename =
            "flow_t" + std::to_string(integrator.getTime()) + ".vtk";
        io::exportToVtk(filename, *ns_solver);
        std::cout << "    Output: " << filename << std::endl;
      }
    }

    step++;
  }

  std::cout << "\n[Complete] Simulation finished successfully!" << std::endl;
  std::cout << "  Total steps: " << integrator.getStepCount() << std::endl;
  std::cout << "  Final time: " << integrator.getTime() << " s" << std::endl;

  // Final output
  io::exportToVtk("flow_final.vtk", *ns_solver);
  std::cout << "  Final state exported to: flow_final.vtk" << std::endl;

  return 0;
}
