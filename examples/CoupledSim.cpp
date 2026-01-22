#include "FEM/NavierStokes.hpp"
#include "FEM/SolidMechanics.hpp"
#include "IO/VtkExport.hpp"
#include "Mesh/Mesh.hpp"
#include "Numerics/CoupledSolver.hpp"
#include <iostream>
#include <memory>

int main() {

  std::cout << "========================================" << std::endl;
  std::cout << "  Coupled Example - Blood Vessel" << std::endl;
  std::cout << "========================================\n" << std::endl;

  // Vessel dimensions (scaled to maintain 15:1 length:radius ratio)
  // Represents 1500 μm length x 100 μm radius vessel
  double vessel_length = 30.0; // scaled units
  double vessel_radius = 2.0;  // scaled units

  // Scale factor for VTK export (converts to micrometers)
  double vtk_scale = 50.0; // 1 sim unit = 50 μm

  // generate mesh (element size for stability)
  auto mesh_ptr = std::make_shared<Mesh>(Mesh::fromPolyhedron(
      Polyhedron::cylendar(vessel_length, vessel_radius, 35), 0.03));
  // setup boundaries
  Mesh::setupBoundaryConditions(Eigen::Vector3d(1, 0, 0), 1, *mesh_ptr);

  // blood properties (for small vessels)
  Fluid blood;
  blood.density = 1060.0; // kg/m³
  blood.turbuelence_model =
      TurbulenceModel::Laminar; // Laminar flow in small vessels
  blood.viscosity_model = ViscosityModel::Newtonian;
  blood.viscosity = 0.0035; // Pa·s (3.5 cP, typical for blood)

  // vessel wall properties (soft tissue)
  double wall_density = 1060.0; // kg/m³
  double y_modulus = 6e4;       // 60 kPa (soft vessel wall)
  double p_ratio = 0.49;        // Nearly incompressible

  Material vessel_wall =
      Material::linear_elastic(y_modulus, p_ratio, wall_density);
  vessel_wall = vessel_wall.with_damping(2.0, 0.005); // Viscoelastic damping

  // cell parameters
  double cell_radius = 1.0; // scaled to match vessel (represents ~50 μm)
  int iso_recursion = 2;
  double Kv = 5.0;                // Volume constraint
  double Ka = 1.0;                // Area constraint
  double Kb = 0.0;                // Bending
  DeformableParticle::Ks = 10.0;  // Stretching stiffness
  DeformableParticle::Kre = 20.0; // Repulsion
  DeformableParticle::Kat = 0.3;  // attraction

  // make cell array
  std::vector<DeformableParticle> cells;
  int n_cells = 64;
  cells.reserve(n_cells);
  for (int i = 0; i < n_cells; i++) {
    cells.emplace_back(iso_recursion, cell_radius);
  }
  for (auto &c : cells) {
    c.setKa(Ka);
    c.setKb(Kb);
    c.setKv(Kv);
  }

  // init coupled solver
  auto solver = CoupledSolver(mesh_ptr);
  solver.initializeDPMSolver(cells);

  // initial cell relaxation / seeding to ECM
  solver.setDPMSolverDt(0.01);
  for (int i = 0; i < 250; i++)
    solver.dpmStep();

  solver.setFluid(blood);
  solver.initializeNavierStokesSolver();
  // Inlet velocity scaled appropriately (represents ~5 mm/s in real vessel)
  solver.setInletVelocity(Eigen::Vector3d(0.1, 0, 0));
  solver.setOutletPressure(0.0);
  solver.setMaterial(vessel_wall);
  solver.initializeSolidMechanicsSolver();
  // Time steps for stable simulation
  // This needs some work as cells move WAYY slower than the other solvers.
  // In reality we'd want cell movements to be at 1e-4 mm/s
  // So... TODO: we can either have Mechanics and Fluid solver for thousands
  // more steps without rebuilding from cell positions. ORRR, we can solve for
  // cell positions with really small dt each step.

  solver.setDPMSolverDt(0.005);
  solver.setFluidSolverDt(0.001);
  solver.setMechanicsDt(0.001);

  int n_steps = 20;
  int n_out = 250;
  for (int i = 0; i < n_out; i++) {
    std::cout << "========  [Main] OUTPUT STEP " << i
              << " ========" << std::endl;
    std::string file_ext = "_" + std::to_string(i) + ".vtk";
    if (solver.getMechanicsSolver())
      io::exportToVtk("solid" + file_ext, *solver.getMechanicsSolver(),
                      vtk_scale);
    if (solver.getFluidSolver())
      io::exportToVtk("fluid" + file_ext, *solver.getFluidSolver(), vtk_scale);
    if (solver.getDPMSolver())
      io::exportToVtk("cells" + file_ext, *solver.getDPMSolver(), vtk_scale);
    for (int j = 0; j < n_steps; j++) {
      solver.integrateStep();
    }
  }

  std::cout << "[Main] Simulation complete." << std::endl;
  return 0;
}
