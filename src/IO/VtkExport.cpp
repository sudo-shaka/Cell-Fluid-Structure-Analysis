#include "DPM/ParticleInteractions.hpp"
#include "FEM/NavierStokes.hpp"
#include "FEM/SolidMechanics.hpp"
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
    const Eigen::Vector3d &p = polyhedron.getPosition(pi);
    out << p(0) << " " << p(1) << " " << p(2) << "\n";
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
    out << p(0) << " " << p(1) << " " << p(2) << "\n";
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
                     const ParticleInteractions &particles) {
  std::ofstream out(filename);
  if (!out.is_open())
    return;

  out << "# vtk DataFile Version 3.0\n";
  out << "Shape export\n";
  out << "ASCII\n";
  out << "DATASET POLYDATA\n";

  int n_faces = 0;
  int n_verts = 0;
  for (size_t i = 0; i < particles.nParticles(); i++) {
    const auto &geo = particles.getParticle(i).getGeometry();
    n_faces += geo.nFaces();
    n_verts += geo.nVerts();
  }

  // Write POINTS
  out << "POINTS " << n_verts << " double\n";

  for (size_t i = 0; i < particles.nParticles(); i++) {
    const auto &geo = particles.getParticle(i).getGeometry();
    for (size_t j = 0; j < geo.nVerts(); j++) {
      const auto &p = geo.getPosition(j);
      out << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
  }

  // Count total entries in face list
  // Each face is a triangle (3 vertices) + 1 count = 4 entries
  size_t totalIndices = n_faces * 4;

  // Write POLYGONS
  out << "POLYGONS " << n_faces << " " << totalIndices << "\n";
  int offset = 0;
  for (size_t pi = 0; pi < particles.nParticles(); pi++) {
    const auto &geo = particles.getParticle(pi).getGeometry();
    for (size_t fi = 0; fi < geo.nFaces(); fi++) {
      const auto &face = geo.getFaceIndices(fi);
      out << "3 " << face[0] + offset << " " << face[1] + offset << " "
          << face[2] + offset << "\n";
    }
    offset += geo.nVerts();
  }

  // Write force vectors
  out << "POINT_DATA " << n_verts << "\n";
  // total forces
  out << "VECTORS total_forces double\n";
  for (size_t i = 0; i < particles.nParticles(); i++) {
    const auto &forces = particles.getParticle(i).getTotalForces();
    for (const auto &f : forces) {
      out << f.x() << " " << f.y() << " " << f.z() << "\n";
    }
  }

  // Volume forces
  out << "VECTORS volume_forces double\n";
  for (size_t i = 0; i < particles.nParticles(); i++) {
    const auto &forces = particles.getParticle(i).getVolumeForces();
    for (const auto &f : forces) {
      out << f.x() << " " << f.y() << " " << f.z() << "\n";
    }
  }

  // Surface area forces
  out << "VECTORS area_forces double\n";
  for (size_t i = 0; i < particles.nParticles(); i++) {
    const auto &forces = particles.getParticle(i).getAreaForces();
    for (const auto &f : forces) {
      out << f.x() << " " << f.y() << " " << f.z() << "\n";
    }
  }

  // Bending forces
  out << "VECTORS bending_forces double\n";
  for (size_t i = 0; i < particles.nParticles(); i++) {
    const auto &forces = particles.getParticle(i).getBendingForces();
    for (const auto &f : forces) {
      out << f.x() << " " << f.y() << " " << f.z() << "\n";
    }
  }

  // Surface adhesion forces (cell to ECM/mesh)
  out << "VECTORS adhesion_forces double\n";
  for (size_t i = 0; i < particles.nParticles(); i++) {
    const auto &forces = particles.getParticle(i).getMatrixAdhesionForces();
    for (const auto &f : forces) {
      out << f.x() << " " << f.y() << " " << f.z() << "\n";
    }
  }

  // Cell-cell attraction forces
  out << "VECTORS attraction_forces double\n";
  for (size_t i = 0; i < particles.nParticles(); i++) {
    const auto &forces = particles.getParticle(i).getCellAdhesionForces();
    for (const auto &f : forces) {
      out << f.x() << " " << f.y() << " " << f.z() << "\n";
    }
  }

  // Cell-cell repulsive forces
  out << "VECTORS repulsive_forces double\n";
  for (size_t i = 0; i < particles.nParticles(); i++) {
    const auto &forces = particles.getParticle(i).getCellRepulsiveForces();
    for (const auto &f : forces) {
      out << f.x() << " " << f.y() << " " << f.z() << "\n";
    }
  }

  // Write shear stress vector
  out << "VECTORS shear_stress double\n";
  for (size_t i = 0; i < particles.nParticles(); i++) {
    const auto &forces = particles.getParticle(i).getShearForces();
    for (const auto &f : forces) {
      out << f.x() << " " << f.y() << " " << f.z() << "\n";
    }
  }

  out << "VECTORS pressure_forces double\n";
  for (size_t i = 0; i < particles.nParticles(); i++) {
    const auto &forces = particles.getParticle(i).getPressureForces();
    for (const auto &f : forces) {
      out << f.x() << " " << f.y() << " " << f.z() << "\n";
    }
  }

  out << "SCALARS vert_types int\n";
  out << "LOOKUP_TABLE default\n";
  for (size_t i = 0; i < particles.nParticles(); i++) {
    const auto &p = particles.getParticle(i);
    for (size_t vi = 0; vi < p.getGeometry().nVerts(); vi++) {
      const auto &vm = p.getVertexMetaData(vi);
      int id = 0;
      if (vm.is_focal_adhesion)
        id += 2;
      if (vm.is_junction)
        id += 1;
      out << id << "\n";
    }
  }
  out << "SCALARS cell_id int\n";
  out << "LOOKUP_TABLE default\n";
  for (size_t i = 0; i < particles.nParticles(); i++) {
    for (size_t j = 0; j < particles.getParticle(i).getGeometry().nVerts(); j++)
      out << i << "\n";
  }
}

void io::exportToVtk(const std::string &filename,
                     const NavierStokesSolver &ns_solver) {
  // First export the mesh (points, cells, and basic point data)
  const Mesh mesh = ns_solver.getMesh();
  io::exportToVtk(filename, mesh);

  // Append solver-specific fields to the same VTK file.
  std::ofstream out(filename, std::ios::app);
  if (!out.is_open()) {
    std::cerr << "[IO] Failed to open VTK file for appending: " << filename
              << std::endl;
    return;
  }

  // Pressure is defined on P1 vertices (mesh.nVertices())
  out << "\nSCALARS pressure double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (size_t i = 0; i < mesh.nVertices(); i++) {
    const double p = ns_solver.getPressureAtNode(i);
    out << p << "\n";
  }

  // Export velocity at P1 vertices (first nVertices entries of solver)
  out << "\nVECTORS velocity double\n";
  for (size_t i = 0; i < mesh.nVertices(); i++) {
    const auto &u = ns_solver.getVelocityAtNode(i);
    out << u(0) << " " << u(1) << " " << u(2) << "\n";
  }

  out.close();
}

void io::exportToVtk(const std::string &filename,
                     const SolidMechanicsSolver &solid_solver) {
  // Export mesh first
  const Mesh &mesh = *solid_solver.getMeshPtr();
  io::exportToVtk(filename, mesh);

  // Append solver-specific fields
  std::ofstream out(filename, std::ios::app);
  if (!out.is_open()) {
    std::cerr << "[IO] Failed to open VTK file for appending: " << filename
              << std::endl;
    return;
  }

  // Export displacement field
  const auto &displacements = solid_solver.getTotalDisplacement();
  out << "\nVECTORS displacement double\n";
  for (const auto &d : displacements) {
    out << d(0) << " " << d(1) << " " << d(2) << "\n";
  }

  // Export velocity field
  const auto &velocities = solid_solver.getVlocity();
  out << "\nVECTORS velocity double\n";
  for (const auto &v : velocities) {
    out << v(0) << " " << v(1) << " " << v(2) << "\n";
  }

  // Export von Mises stress
  const auto &von_mises = solid_solver.getVonMisesStress();
  out << "\nSCALARS von_mises_stress double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (double vm : von_mises) {
    out << vm << "\n";
  }

  // Export strain tensor (as 6 components: xx, yy, zz, xy, xz, yz)
  const auto &strains = solid_solver.getStrain();
  out << "\nSCALARS strain_xx double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (const auto &s : strains) {
    out << s(0, 0) << "\n";
  }

  out << "\nSCALARS strain_yy double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (const auto &s : strains) {
    out << s(1, 1) << "\n";
  }

  out << "\nSCALARS strain_zz double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (const auto &s : strains) {
    out << s(2, 2) << "\n";
  }

  // Export stress tensor components
  const auto &stresses = solid_solver.getStress();
  out << "\nSCALARS stress_xx double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (const auto &s : stresses) {
    out << s(0, 0) << "\n";
  }

  out << "\nSCALARS stress_yy double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (const auto &s : stresses) {
    out << s(1, 1) << "\n";
  }

  out << "\nSCALARS stress_zz double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (const auto &s : stresses) {
    out << s(2, 2) << "\n";
  }

  out.close();
}
