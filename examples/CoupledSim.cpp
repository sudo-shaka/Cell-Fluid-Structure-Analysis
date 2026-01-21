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

  // generate mesh
  auto mesh_ptr = std::make_shared<Mesh>(
      Mesh::fromPolyhedron(Polyhedron::cylendar(25, 2, 30), 0.2));
  // setup boundaries
  Mesh::setupBoundaryConditions(Eigen::Vector3d(1, 0, 0), 1, *mesh_ptr);

  // blood properties
  Fluid blood;
  blood.density = 1000.0;
  blood.turbuelence_model = TurbulenceModel::Laminar;
  blood.viscosity_model = ViscosityModel::Newtonian;
  blood.viscosity = 0.003;

  // vessel wall properties
  double wall_density = 1100.0;
  double y_modulus = 1e6;
  double p_ratio = 0.46;

  Material vessel_wall =
      Material::linear_elastic(y_modulus, p_ratio, wall_density);
  vessel_wall = vessel_wall.with_damping(0.01, 0.001);

  // cell parameters
  double cell_radius = 1.0;
  int iso_recursion = 2;
  double Kv = 5.0;
  double Ka = 1.0;
  double Kb = 0.0;
  DeformableParticle::Ks = 7.0;
  DeformableParticle::Kre = 20.0;

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
  solver.setFluid(blood);
  solver.initializeNavierStokesSolver();
  solver.setInletVelocity(Eigen::Vector3d(1.0, 0, 0));
  solver.setOutletPressure(0.0);
  solver.setMaterial(vessel_wall);
  solver.initializeSolidMechanicsSolver();
  solver.setDPMSolverDt(0.01);
  solver.setFluidSolverDt(0.001);
  solver.setMechanicsDt(0.001);

  int n_steps = 20;
  int n_out = 50;
  for (int i = 0; i < n_out; i++) {
    std::cout << "========  [Main] OUTPUT STEP " << i
              << " ========" << std::endl;
    std::string file_ext = "_" + std::to_string(i) + ".vtk";
    if (solver.getMechanicsSolver())
      io::exportToVtk("solid" + file_ext, *solver.getMechanicsSolver());
    if (solver.getFluidSolver())
      io::exportToVtk("fluid" + file_ext, *solver.getFluidSolver());
    if (solver.getDPMSolver())
      io::exportToVtk("cells" + file_ext, *solver.getDPMSolver());
    for (int j = 0; j < n_steps; j++) {
      solver.integrateStep();
    }
  }

  std::cout << "[Main] Simulation complete." << std::endl;
  return 0;
}
