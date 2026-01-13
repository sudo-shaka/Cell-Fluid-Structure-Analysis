#include "BC/BC.hpp"
#include "Mesh/Mesh.hpp"
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <iostream>

void boundary_assignment::setupBoundaryConditions(
    const glm::dvec3 &inlet_to_outlet_direction, Mesh &mesh,
    const double percent_inlet_outlet_converage) {
  if (mesh.nFaces() == 0 || mesh.nVertices() < 3) {
    return;
  }

  std::cout << "[BC] Applying boundary conditions to mesh... " << std::flush;

  int n_wall = 0;
  int n_inlet = 0;
  int n_outlet = 0;

  glm::dvec3 min_bounds = mesh.getVertexPositon(0);
  glm::dvec3 max_bounds = mesh.getVertexPositon(0);

  for (size_t vi = 0; vi < mesh.nVertices(); vi++) {
    min_bounds = glm::min(min_bounds, mesh.getVertexPositon(vi));
    max_bounds = glm::max(max_bounds, mesh.getVertexPositon(vi));
  }

  const glm::dvec3 mesh_size = max_bounds - min_bounds;
  int primary_axis = 0;
  if (mesh_size.y > mesh_size.x && mesh_size.y > mesh_size.z)
    primary_axis = 1;
  else if (mesh_size.z > mesh_size.x && mesh_size.z > mesh_size.y)
    primary_axis = 2;

  // Boundary assignment params
  const double tolerance =
      mesh_size[primary_axis] * 0.01 * percent_inlet_outlet_converage;
  ;
  const double inlet_pos = inlet_to_outlet_direction[primary_axis] > 0
                               ? min_bounds[primary_axis]
                               : max_bounds[primary_axis];
  const double outlet_pos = inlet_to_outlet_direction[primary_axis] > 0
                                ? max_bounds[primary_axis]
                                : min_bounds[primary_axis];

  for (size_t vi = 0; vi < mesh.nVertices(); vi++) {
    // set all boundaries to internal.
    mesh.setFluidVertexBC(vi, FluidBCType::Internal);
    mesh.setSolidVertexBC(vi, SolidBCType::Free);
  }

  const std::vector<Face> &faces = mesh.getFaces();
  for (size_t fi = 0; fi < mesh.nFaces(); fi++) {
    if (mesh.isFaceInternal(fi)) {
      continue; // skip internal faces
    }
    // if wall, set to wall
    for (const int &vi : faces[fi].vertids) {
      mesh.setFluidVertexBC(vi, FluidBCType::Wall);
      n_wall++;
    }

    // assign inlet and outlet based on position and face normal with respect to
    // flow direction

    double dist_to_inlet = std::abs(faces[fi].center[primary_axis] - inlet_pos);
    double dist_to_outlet =
        std::abs(faces[fi].center[primary_axis] - outlet_pos);

    if (dist_to_inlet < tolerance) {
      for (const auto &vi : faces[fi].vertids) {
        mesh.setFluidVertexBC(vi, FluidBCType::Inlet);
        mesh.setSolidVertexBC(vi, SolidBCType::Fixed);
        n_inlet++;
        n_wall--;
      }
    } else if (dist_to_outlet < tolerance) {
      for (const int &vi : faces[fi].vertids) {
        mesh.setFluidVertexBC(vi, FluidBCType::Outlet);
        mesh.setSolidVertexBC(vi, SolidBCType::Fixed);
        n_outlet++;
        n_wall--;
      }
    }
  }
  // set p2 data
  mesh.setP2BoundariesFromP1Boundaries();
  std::cout << "Competed.\n";
  std::cout << "[BC] NavierStokes boundaries:\n";
  std::cout << "\tInlet BC:" << n_inlet << " Outlet BC:" << n_outlet
            << " Wall BC:" << n_wall << "\n";
  std::cout << "[BC] Solid mechanic boundaries:\n";
  std::cout << "\tFixed BC:" << n_outlet + n_inlet
            << " Free BC:" << mesh.nVertices() - n_outlet - n_inlet
            << std::endl;
}
