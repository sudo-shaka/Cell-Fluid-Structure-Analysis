#include "FEM/NavierStokes.hpp"
#include "IO/VtkExport.hpp"
#include "Mesh/Mesh.hpp"
#include "Numerics/FEMIntegrator.hpp"
#include <iostream>
#include <memory>
#include <string>

int main() {
  std::cout << "=== Navier-Stokes Flow Simulation ===" << std::endl;

  // Flow parameters (reduced for laminar regime)
  const double inlet_velocity = 1.0;  // m/s
  const double outlet_pressure = 0.0; // Pa (gauge)

  // Fluid properties (water at 20°C)
  const double density = 1000.0;  // kg/m³
  const double viscosity = 0.003; // Pa·s

  // Time integration parameters
  const double dt = 1e-6;
  const double total_time = 5.0;
  const int output_interval = 10;

  double radius = 2.0;
  double length = 50.0;
  int n_surface_points = 30;
  double max_edge_length = 0.1;

  auto mesh = std::make_shared<Mesh>(Mesh::fromPolyhedron(
      Polyhedron::cylendar(length, radius, n_surface_points), max_edge_length));
  mesh = std::make_shared<Mesh>(
      Mesh::fromObjFile("../meshes/split_merge.obj", 50.5));

  std::cout << "[Setup] Mesh created with " << mesh->nVertices()
            << " vertices, " << mesh->nTets() << " tetrahedra" << std::endl;

  // Setup boundary conditions
  Eigen::Vector3d flow_direction(1, 0, 0);
  Mesh::setupBoundaryConditions(flow_direction, 1, *mesh);

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
  ns_solver->setInletType(InletType::Uniform);
  ns_solver->setRelaxU(0.5);
  ns_solver->setRelaxP(0.5);
  std::cout << "[Setup] Fluid properties:" << std::endl;
  std::cout << "  Density: " << density << " kg/m³" << std::endl;
  std::cout << "  Viscosity: " << viscosity << " Pa·s" << std::endl;
  std::cout << "  Inlet velocity: " << inlet_velocity << " m/s" << std::endl;

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

    //  Periodic output
    if (step % output_interval == 0) {
      // Get max velocity and pressure
      double max_vel = 0.0;
      double min_pressure = 1e9, max_pressure = -1e9;
      double avg_vx = 0.0, avg_vy = 0.0, avg_vz = 0.0;
      const auto &velocities = ns_solver->getVelocity();
      const auto &pressures = ns_solver->getPressure();

      for (const auto &v : velocities) {
        max_vel = std::max(max_vel, v.norm());
        avg_vx += v.x();
        avg_vy += v.y();
        avg_vz += v.z();
      }
      avg_vx /= velocities.size();
      avg_vy /= velocities.size();
      avg_vz /= velocities.size();

      for (const auto &p : pressures) {
        min_pressure = std::min(min_pressure, p);
        max_pressure = std::max(max_pressure, p);
      }

      std::cout << "  Step " << step << ": t = " << integrator.getTime()
                << " s, max velocity = " << max_vel << " m/s" << std::endl;
      std::cout << "    Avg velocity: (" << avg_vx << ", " << avg_vy << ", "
                << avg_vz << ")" << std::endl;
      std::cout << "    Pressure range: [" << min_pressure << ", "
                << max_pressure << "] Pa" << std::endl;

      // Export VTK for visualization
      if (step % (output_interval * 5) == 0) {
        std::string filename = "flow_t" + std::to_string(step) + ".vtk";
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
