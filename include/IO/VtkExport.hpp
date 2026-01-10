#pragma once

#include <string>
class Polyhedron;
class Mesh;
class DeformableParticle;
class ParticleInteractions;

namespace io {
void exportToVtk(const std::string &filename, const Polyhedron &polyhedron);
void exportToVtk(const std::string &filename, const Mesh &mesh);
void exportToVtk(const std::string &filename, const DeformableParticle &dp);
void exportToVtk(const std::string &filename,
                 const ParticleInteractions &particles);
}; // namespace  io
