#include "DPM/DeformableParticle.hpp"
#include "Polyhedron/Polyhedron.hpp"
#include <cstdlib>
#include <string>
#include <cmath>
#include <iostream>
#include "IO/VtkExport.hpp"

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path-to-obj>\n";
    return 1;
  }

  std::string obj_path = argv[1];
  if (!obj_path.empty() && obj_path[0] == '~') {
    const char *home = std::getenv("HOME");
    if (home != nullptr) {
      obj_path = std::string(home) + obj_path.substr(1);
    }
  }

  Polyhedron p = Polyhedron::fromWavefront(obj_path);
  io::exportToVtk("vesselbed_polyhedron.vtk", p);
  return 0;
}
