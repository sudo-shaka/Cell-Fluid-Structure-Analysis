#pragma once

#include <string>
class Polyhedron;
class Mesh;
class DeformableParticle;
class NavierStokesSolver;
class SolidMechanicsSolver;
class ParticleInteractions;

namespace io {
void exportToVtk(const std::string &filename, const Polyhedron &polyhedron);
void exportToVtk(const std::string &filename, const Mesh &mesh);
void exportToVtk(const std::string &filename, const DeformableParticle &dp);
void exportToVtk(const std::string &filename,
                 const ParticleInteractions &particles);
void exportToVtk(const std::string &filename,
                 const NavierStokesSolver &ns_solver);
void exportToVtk(const std::string &filename,
                 const SolidMechanicsSolver &solid_mechanics);
}; // namespace io

// Static utility class for easier access
class VtkExport {
public:
  static void exportMesh(const Mesh &mesh, const std::string &filename) {
    io::exportToVtk(filename, mesh);
  }

  static void
  exportMeshWithVelocityAndPressure(const Mesh &mesh,
                                    const NavierStokesSolver &solver,
                                    const std::string &filename) {
    io::exportToVtk(filename, solver);
  }

  static void
  exportMeshWithDisplacementAndStress(const Mesh &mesh,
                                      const SolidMechanicsSolver &solver,
                                      const std::string &filename) {
    io::exportToVtk(filename, solver);
  }
};
