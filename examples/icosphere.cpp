#include "DPM/DeformableParticle.hpp"
#include <cmath>
#include <iostream>

int main() {
  // create an isosphere of radius 1 with recursion level 2
  double radius = 1.0;
  Polyhedron p = Polyhedron::isosphere(radius, 3);

  double ideal_surface_area = 4.0 * M_PI * std::pow(radius, 2.0);
  double ideal_volume = (4.0 / 3.0) * M_PI * std::pow(radius, 3.0);

  std::cout << "Vertices: " << p.nVerts() << "\n";
  std::cout << "Faces: " << p.nFaces() << "\n";
  std::cout << "Ideal Volume: " << ideal_volume << "\n";
  std::cout << "Volume: " << p.getVolume() << "\n";
  std::cout << "Ideal Surface area: " << ideal_surface_area << "\n";
  std::cout << "Surface area: " << p.getSurfaceArea() << "\n";
  const auto &c = p.getCentroid();
  std::cout << "Centroid: " << c.x() << ", " << c.y() << ", " << c.z() << "\n";

  return 0;
}
