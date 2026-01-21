#include "FEM/NavierStokes.hpp"
#include "FEM/SolidMechanics.hpp"
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
      Mesh::fromPolyhedron(Polyhedron::cylendar(25, 2, 30)));
  // setup boundaries
  Mesh::setupBoundaryConditions(Eigen::Vector3d(1, 0, 0), 1, *mesh_ptr);

  // blood properties
  Fluid blood;
  blood.density = 1000.0;
  blood.turbuelence_model = TurbulenceModel::Laminar;
  blood.viscosity_model = ViscosityModel::Newtonian;
  blood.viscosity = 0.003;

  // vessel wall properties
  Material vessel_wall = Material::linear_elastic(1e3, 0.45, 1100.0);
  vessel_wall = vessel_wall.with_damping(0.01, 0.001);

  // cell parameters
  double cell_radius = 1.0;
  int iso_recursion = 2;
  double Kv = 5.0;
  double Ka = 2.0;
  double Kb = 0.0;
  DeformableParticle::Ks = 7.0;
  DeformableParticle::Kre = 20.0;

  // make cell array
  std::vector<DeformableParticle> cells;
  int n_cells = 32;
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
  solver.setMaterial(vessel_wall);
  solver.initializeSolidMechanicsSolver();

  return 1;
}
