#pragma once

#include <string>
class Polyhedron;
class Mesh;
class DeformableParticle;
class NavierStokesSolver;
class SolidMechanicsSolver;
class ParticleInteractions;

namespace io {
void exportToVtk(const std::string &filename, const Polyhedron &polyhedron,
                 double scale = 1.0);
void exportToVtk(const std::string &filename, const Mesh &mesh,
                 double scale = 1.0);
void exportToVtk(const std::string &filename, const DeformableParticle &dp,
                 double scale = 1.0);
void exportToVtk(const std::string &filename,
                 const ParticleInteractions &particles, double scale = 1.0);
void exportToVtk(const std::string &filename,
                 const NavierStokesSolver &ns_solver, double scale = 1.0);
void exportToVtk(const std::string &filename,
                 const SolidMechanicsSolver &solid_mechanics,
                 double scale = 1.0);
}; // namespace io
