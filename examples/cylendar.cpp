#include "IO/VtkExport.hpp"
#include "Polyhedron/Polyhedron.hpp"

int main() {
  // build a cylinder polyhedron: length=2.0, radius=1.0, resolution=24
  double length = 10.0;
  double radius = 1.0;
  int resolution = 50;

  Polyhedron poly = Polyhedron::cylendar(length, radius, resolution);
  io::exportToVtk("test_cylendar.vtk", poly);

  return 0;
}
