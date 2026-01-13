#include "IO/VtkExport.hpp"
#include "Mesh/Mesh.hpp"
#include "Polyhedron/Polyhedron.hpp"
#include <fstream>
#include <iostream>

// helper function
void exportVtkHeader(std::ofstream &out, const std::string &dataset) {
  out << "# vtk DataFile Version 3.0\n";
  out << "Shape export\n";
  out << "ASCII\n";
  out << "DATASET " << dataset << "\n";
}

void io::exportToVtk(const std::string &filename,
                     const Polyhedron &polyhedron) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    std::cerr << "[IO] Failed to open file: " << filename << std::endl;
    return;
  }

  exportVtkHeader(out, "POLYDATA");
  size_t n_verts = polyhedron.nVerts();
  out << "POINTS " << n_verts << " double\n";
  for (size_t pi = 0; pi < n_verts; pi++) {
    const glm::dvec3 &p = polyhedron.getPosition(pi);
    out << p.x << " " << p.y << " " << p.z << "\n";
  }

  size_t n_faces = polyhedron.nFaces();
  size_t total_inds = 0;
  for (size_t fi = 0; fi < n_faces; fi++) {
    const auto &face = polyhedron.getFaceIndices(fi);
    total_inds += face.size() + 1;
  }
  out << "POLYGONS " << n_faces << " " << total_inds << "\n";
  for (size_t i = 0; i < n_faces; i++) {
    const auto &face = polyhedron.getFaceIndices(i);
    out << face.size();
    for (int idx : face)
      out << " " << idx;
    out << "\n";
  }
}
void io::exportToVtk(const std::string &filename, const Mesh &mesh) {
  std::ofstream out(filename);
  const auto &tets = mesh.getTets();
  const auto &positions = mesh.getVertPositions();
  if (!out.is_open()) {
    std::cerr << "[IO] Failed to open file: " << filename << std::endl;
    return;
  }
  exportVtkHeader(out, "UNSTRUCTURED_GRID");
  out << "POINTS " << positions.size() << " double\n";
  for (const auto &p : positions) {
    out << p.x << " " << p.y << " " << p.z << "\n";
  }
  size_t n_tets = tets.size();
  out << "CELLS " << n_tets << " " << n_tets * 5 << "\n";
  for (const auto &tet : tets) {
    out << "4 " << tet.vertids[0] << " " << tet.vertids[1] << " "
        << tet.vertids[2] << " " << tet.vertids[3] << "\n";
  }

  out << "CELL_TYPES " << n_tets << "\n";
  for (size_t i = 0; i < n_tets; ++i) {
    out << "10\n";
  }

  out << "\nPOINT_DATA " << positions.size() << "\n";
  out << "SCALARS FluidBoundaryType int 1\n";
  out << "LOOKUP_TABLE default\n";

  for (size_t vi = 0; vi < positions.size(); vi++) {
    int bc = static_cast<int>(mesh.getFluidVertexBC(vi));
    out << bc << "\n";
  }

  out << "SCALARS SolidBoundaryType int 1\n";
  out << "LOOKUP_TABLE default\n";
  for (size_t vi = 0; vi < positions.size(); vi++) {
    int bc = static_cast<int>(mesh.getSolidVertexBC(vi));
    out << bc << "\n";
  }
}
void io::exportToVtk(const std::string &filename,
                     const DeformableParticle &dp) {}
void io::exportToVtk(const std::string &filename,
                     const ParticleInteractions &particles) {}
