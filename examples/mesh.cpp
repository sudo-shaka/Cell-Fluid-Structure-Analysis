#include "Mesh/Mesh.hpp"
#include "Polyhedron/Polyhedron.hpp"
#include <iostream>

int main() {
  // build a cylinder polyhedron: length=2.0, radius=1.0, resolution=24
  double length = 10.0;
  double radius = 1.0;
  int resolution = 50;

  Polyhedron poly = Polyhedron::cylendar(length, radius, resolution);

  std::cout << "Polyhedron vertices: " << poly.nVerts() << " faces: " << poly.nFaces() << "\n";
  size_t nprintv = std::min<size_t>(6, poly.nVerts());
  for (size_t i = 0; i < nprintv; ++i) {
    const auto &pv = poly.getPosition(i);
    std::cout << "v[" << i << "] = (" << pv.x << ", " << pv.y << ", " << pv.z << ")\n";
  }

  // create a mesh from the polyhedron with a target max edge length
  double max_edge_length = 0.2;
  Mesh mesh = Mesh::fromPolyhedron(poly, max_edge_length);

  std::cout << "Mesh initialized: " << (mesh.isInitialized() ? "yes" : "no") << "\n";
  std::cout << "Vertices (P1): " << mesh.nVertices() << "\n";
  std::cout << "Tetrahedra: " << mesh.nTets() << "\n";
  std::cout << "Faces: " << mesh.nFaces() << "\n";

  if (mesh.isInitialized()) {
    // Print a few tet volumes and centroids
    size_t nprint = std::min<size_t>(5, mesh.nTets());
    for (size_t i = 0; i < nprint; ++i) {
      const auto &t = mesh.tetAt(i);
      std::cout << "Tet " << i << " volume: " << t.volume << " centroid: (" \
                << t.centroid.x << ", " << t.centroid.y << ", " << t.centroid.z << ")\n";
    }
  }

  return 0;
}
