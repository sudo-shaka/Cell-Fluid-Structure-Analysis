#include "Mesh/Mesh.hpp"
#include "IO/VtkExport.hpp"
#include "Polyhedron/Polyhedron.hpp"

int main() {
  // build a cylinder polyhedron: length=2.0, radius=1.0, resolution=24
  double length = 10.0;
  double radius = 1.0;
  int resolution = 50;

  Polyhedron poly = Polyhedron::cylendar(length, radius, resolution);
  // create a mesh from the polyhedron with a target max edge length
  double max_edge_length = 0.01;
  Mesh mesh = Mesh::fromPolyhedron(poly, max_edge_length);
  Mesh::setupBoundaryConditions(Eigen::Vector3d{1, 0, 0}, 5.0, mesh);

  io::exportToVtk("test_mesh.vtk", mesh);

  return 0;
}
