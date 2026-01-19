#include "DPM/DeformableParticle.hpp"
#include "DPM/ParticleInteractions.hpp"
#include "IO/VtkExport.hpp"
#include "Mesh/Mesh.hpp"
#include "Numerics/DPMIntegrator.hpp"
#include <iostream>
#include <memory>

int main() {
  std::cout << "=== Deformable Particle Model (DPM) Tissue Simulation ==="
            << std::endl;

  // Simulation parameters
  const int num_cells = 30;         // Number of cells in tissue
  const double dt = 0.0001;         // Time step size (reduced for stability)
  const double total_time = 0.1;    // Total simulation time
  const int output_interval = 100;  // Output every N steps
  const int recursion_level = 2;    // Icosphere recursion for cell geometry

  // Cell mechanical properties
  const double cell_radius = 0.3;       // Initial cell radius
  const double shape_param = 1.0;       // Ideal shape parameter (calA0)
  const double Kv = 1.0;                // Volume stiffness (reduced)
  const double Ka = 0.5;                // Surface area stiffness (reduced)
  const double Kb = 0.01;               // Bending stiffness

  // Interaction parameters (static properties)
  DeformableParticle::Ks = 0.5;   // Cell-Matrix adhesion (reduced)
  DeformableParticle::Kat = 0.5;  // Cell-Cell adhesion (reduced)
  DeformableParticle::Kre = 2.0;  // Cell-Cell repulsion (reduced)

  // Create mesh substrate for cells to adhere to
  std::cout << "\n[Setup] Creating cylindrical mesh substrate..." << std::endl;
  double cylinder_length = 10.0;
  double cylinder_radius = 2.0;
  int n_surface_points = 30;
  double max_edge_length = 0.5;

  auto mesh = std::make_shared<Mesh>(Mesh::fromPolyhedron(
      Polyhedron::cylendar(cylinder_length, cylinder_radius, n_surface_points),
      max_edge_length));

  std::cout << "[Setup] Mesh created with " << mesh->nVertices()
            << " vertices, " << mesh->nFaces() << " faces, "
            << mesh->nTets() << " tetrahedra" << std::endl;

  std::cout << "\n[Setup] Creating tissue with " << num_cells << " cells..."
            << std::endl;

  // Create cells at origin (they will be dispersed to mesh faces)
  std::vector<DeformableParticle> cells;
  cells.reserve(num_cells);

  for (int i = 0; i < num_cells; ++i) {
    Eigen::Vector3d position(0.0, 0.0, 0.0);
    cells.emplace_back(position, shape_param, recursion_level, cell_radius, Kv,
                       Ka, Kb);
  }

  std::cout << "[Setup] Cells created with properties:" << std::endl;
  std::cout << "  Radius: " << cell_radius << std::endl;
  std::cout << "  Volume stiffness (Kv): " << Kv << std::endl;
  std::cout << "  Area stiffness (Ka): " << Ka << std::endl;
  std::cout << "  Bending stiffness (Kb): " << Kb << std::endl;
  std::cout << "  Cell-Cell adhesion (Kat): " << DeformableParticle::Kat
            << std::endl;
  std::cout << "  Cell-Cell repulsion (Kre): " << DeformableParticle::Kre
            << std::endl;

  // Create tissue system with particle interactions
  auto tissue = std::make_shared<ParticleInteractions>(std::move(cells));
  std::cout << "[Setup] Tissue initialized with " << tissue->nParticles()
            << " particles" << std::endl;

  // Disperse cells evenly across mesh surface faces
  std::cout << "[Setup] Dispersing cells to mesh surface faces..." << std::endl;
  tissue->disperseCellsToFaceCenters(mesh->getFaces());
  std::cout << "[Setup] Cells positioned on mesh surfaces" << std::endl;

  // Create time integrator
  DPMTimeIntegrator integrator(tissue);
  integrator.dt = dt;

  std::cout << "\n[Simulation] Starting DPM tissue simulation..." << std::endl;
  std::cout << "  Time step: " << dt << " s" << std::endl;
  std::cout << "  Total time: " << total_time << " s" << std::endl;
  std::cout << "  Expected steps: " << static_cast<int>(total_time / dt)
            << std::endl;

  int step = 0;
  double current_time = 0.0;

  while (current_time < total_time) {
    // Advance one time step
    integrator.advanceStep();

    current_time += dt;

    // Periodic output
    if (step % output_interval == 0) {
      std::cout << "  Step " << step << ": t = " << current_time << " s"
                << std::endl;

      // Export tissue state to VTK for visualization
      if (step % (output_interval * 5) == 0) {
        std::string filename =
            "dpm_tissue_t" + std::to_string(current_time) + ".vtk";
        io::exportToVtk(filename, *tissue);
        std::cout << "    Output: " << filename << std::endl;
      }
    }

    step++;
  }

  std::cout << "\n[Complete] Simulation finished successfully!" << std::endl;
  std::cout << "  Total steps: " << step << std::endl;
  std::cout << "  Final time: " << current_time << " s" << std::endl;

  // Final output
  io::exportToVtk("dpm_tissue_final.vtk", *tissue);
  std::cout << "  Final state exported to: dpm_tissue_final.vtk" << std::endl;

  return 0;
}
