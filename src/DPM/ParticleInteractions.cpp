#include "DPM/ParticleInteractions.hpp"
#include "Mesh/Mesh.hpp"
#include <algorithm>
#include <cfloat>
#include <iostream>
#include <random>
#include <unordered_set>

void ParticleInteractions::rebuildIntercellularSpatialGrid() {
  spatial_grid_.clear();
  for (size_t ci = 0; ci < particles_.size(); ci++) {
    const auto &shape = particles_[ci].getGeometry();
    for (size_t vi = 0; vi < shape.nVerts(); vi++) {
      spatial_grid_.insert(ci, vi, shape.getPosition(vi));
    }
  }
}
void ParticleInteractions::rebuildMatrixFacesSpatialGrid(
    const std::vector<Face> &faces, SpatialHashGrid &grid) {
  grid.clear();
  for (size_t fi = 0; fi < faces.size(); fi++) {
    grid.insert(0, fi, faces[fi].center);
  }
}
void ParticleInteractions::queryNeighbors(
    const Eigen::Vector3d &pos, double radius,
    std::vector<SpatialHashGrid::CellVertex> &out) const {
  spatial_grid_.queryNeighbors(pos, radius, out);
}

void ParticleInteractions::queryFaceNeighbors(
    const Eigen::Vector3d &pos, double radius, const SpatialHashGrid &grid,
    std::vector<SpatialHashGrid::CellVertex> &out) const {
  grid.queryNeighbors(pos, radius, out);
}

// interaction functions
void ParticleInteractions::disperseCellsToFaceCenters(
    const std::vector<Face> &faces) {
  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  // Filter to only boundary faces (faces with only one adjacent tetrahedron)
  std::vector<size_t> boundary_face_indices;
  std::vector<double> boundary_face_areas;
  double total_area = 0.0;

  for (size_t i = 0; i < faces.size(); i++) {
    // A boundary face has tet_b == -1 (only one adjacent tetrahedron)
    if (faces[i].tet_b == -1) {
      boundary_face_indices.push_back(i);
      boundary_face_areas.push_back(faces[i].area);
      total_area += faces[i].area;
    }
  }

  if (boundary_face_indices.empty()) {
    std::cerr << "[DPM][Warning] No boundary faces found for cell dispersion"
              << std::endl;
    return;
  }

  // Build cumulative probability distribution based on area
  std::vector<double> cum_prob(boundary_face_indices.size());
  double sum = 0.0;
  for (size_t i = 0; i < boundary_face_indices.size(); i++) {
    sum += boundary_face_areas[i] / total_area;
    cum_prob[i] = sum;
  }

  // Place each particle at a randomly selected boundary face center
  for (size_t i = 0; i < particles_.size(); i++) {
    double rand = dist(gen);
    auto it = std::lower_bound(cum_prob.begin(), cum_prob.end(), rand);
    size_t local_index = std::distance(cum_prob.begin(), it);
    size_t face_index = boundary_face_indices[local_index];
    const Eigen::Vector3d &face_center = faces[face_index].center;
    particles_[i].moveTo(face_center);
  }
}
void ParticleInteractions::interactingForceUpdate(const size_t pi) {
  assert(pi < particles_.size());
  if (DeformableParticle::Kre > 1e-12) {
    cellCellRepulsionUpdate(pi);
  }
  if (DeformableParticle::Kat > 1e-12) {
    simpleSpringAttraction(pi);
  }
}
void ParticleInteractions::cellCellRepulsionUpdate(
    const size_t particle_index) {

  if (particles_[particle_index].Kre < 1e-12) {
    return;
  }
  auto &particle = particles_[particle_index];
  const auto &geo = particle.getGeometry();
  const auto com = geo.getCentroid();
  double max_dist = particle.getMaxInteractingDistance();

  std::unordered_set<int> nearby_cells;
  std::vector<SpatialHashGrid::CellVertex> neighbors;
  neighbors.reserve(200);

  for (size_t vi = 0; vi < geo.nVerts(); vi++) {
    const Eigen::Vector3d &position = geo.getPosition(vi);
    queryNeighbors(position, max_dist * 2.0, neighbors);
    for (const auto &neighbor : neighbors) {
      if (neighbor.cell_idx != static_cast<int>(particle_index)) {
        nearby_cells.insert(neighbor.cell_idx);
      }
    }
  }

  for (int pj : nearby_cells) {
    auto windingNumbers =
        findWindingNumbersBetween(particles_[particle_index], particles_[pj]);
    for (size_t vi = 0; vi < geo.nVerts(); vi++) {
      if (windingNumbers[vi] < 1e-8)
        continue;
      Eigen::Vector3d force = (com - geo.getPosition(vi)).normalized();
      force *= DeformableParticle::Kre * windingNumbers[vi];
      particles_[particle_index].addRepulsiveForce(vi, force);
    }
  }
}
void ParticleInteractions::simpleSpringAttraction(const size_t particle_index) {

  std::vector<SpatialHashGrid::CellVertex> neighbors;
  neighbors.reserve(100);

  auto &particle = particles_[particle_index];
  double max_dist = particle.getMaxInteractingDistance();
  double l0 = particle.getRestingEdgeLength();
  assert(l0 >= 1e-12);

  for (size_t vi = 0; vi < particle.getGeometry().nVerts(); vi++) {
    int n_partners = 0;
    Eigen::Vector3d force = Eigen::Vector3d::Zero();
    auto &vert_meta = particle.getMutVertexMeta(vi);
    vert_meta.is_junction = false;
    const auto &vert_position = particle.getGeometry().getPosition(vi);

    queryNeighbors(vert_position, max_dist, neighbors);

    for (const auto &neighbor : neighbors) {
      if (neighbor.cell_idx == static_cast<int>(particle_index))
        continue;
      double dist = (neighbor.position - vert_position).norm();
      if (dist >= max_dist)
        continue;
      ++n_partners;
      force += DeformableParticle::Kat * 0.5 * ((dist / l0) - 1.0) *
               (vert_position - neighbor.position).normalized();
    }
    if (n_partners == 0)
      continue;
    vert_meta.is_junction = true;
    force /= static_cast<double>(n_partners);
    particle.setAttactionForce(vi, force);
  }
}

void ParticleInteractions::interactWithMesh(const Mesh &mesh) {
  const auto &faces = mesh.getFaces();
  // TODO: make parallel
  for (size_t i = 0; i < particles_.size(); i++) {
    cellMeshInteractionUpdate(faces, i);
  }
}

void ParticleInteractions::cellMeshInteractionUpdate(
    const std::vector<Face> &faces, const size_t particle_index) {
  if (DeformableParticle::Ks < 1e-8)
    return;
  assert(particle_index < particles_.size());
  auto &particle = particles_[particle_index];
  const auto &geo = particle.getGeometry();
  double max_dist = particle.getMaxInteractingDistance();
  double l0 = particle.getRestingEdgeLength();
  assert(l0 > 1e-8);

  // Build spatial grid only with boundary faces
  SpatialHashGrid face_grid(max_dist);
  for (size_t fi = 0; fi < faces.size(); fi++) {
    if (faces[fi].tet_b == -1) { // Only boundary faces
      face_grid.insert(0, fi, faces[fi].center);
    }
  }

  std::vector<SpatialHashGrid::CellVertex> nearby_faces;
  nearby_faces.reserve(50);

  for (size_t vi = 0; vi < geo.nVerts(); vi++) {
    auto &vert_meta = particle.getMutVertexMeta(vi);
    vert_meta.is_focal_adhesion = false;
  }

  for (size_t pfaceidx = 0; pfaceidx < geo.nFaces(); pfaceidx++) {
    const auto &p_face = geo.getFaceIndices(pfaceidx);
    int n_partners = 0;
    Eigen::Vector3d force = Eigen::Vector3d::Zero();
    const Eigen::Vector3d &p_face_center = geo.getFaceCentroid(pfaceidx);
    const Eigen::Vector3d &p_face_normal = geo.getFaceNormals(pfaceidx);

    queryFaceNeighbors(p_face_center, max_dist, face_grid, nearby_faces);

    for (const auto &neighbor : nearby_faces) {
      size_t mesh_face_index = neighbor.vertex_idx;
      if (!faces[mesh_face_index].is_ecm)
        continue;
      const Eigen::Vector3d &mesh_face_center = faces[mesh_face_index].center;
      Eigen::Vector3d rij = mesh_face_center - p_face_center;
      double dist2 = rij.squaredNorm();
      if (dist2 > max_dist * max_dist)
        continue;
      const Eigen::Vector3d &mesh_face_normal = -faces[mesh_face_index].normal;
      bool faces_each_other = p_face_normal.dot(mesh_face_normal) > 0.0;
      if (!faces_each_other)
        continue;
      n_partners++;
      const Eigen::Vector3d &com = geo.getCentroid();
      Eigen::Vector3d ftmp = ((std::sqrt(dist2) / l0) - 1.0) *
                             (com - mesh_face_center).normalized();
      force += DeformableParticle::Ks * ftmp;
    }
    if (n_partners == 0)
      continue;
    force /= static_cast<double>(n_partners);
    for (int vi = 0; vi < 3; vi++) {
      particle.addSurfaceAdhesionForce(p_face[vi], force);
      auto &vert_meta = particle.getMutVertexMeta(p_face[vi]);
      vert_meta.is_focal_adhesion = true;
    }
  }
}

void ParticleInteractions::cellAttractToSurface(
    const size_t cellidx, const std::vector<Face> &faces) {
  if (DeformableParticle::Ks < 1e-8) {
    return;
  }
  if (cellidx >= particles_.size()) {
    std::cerr << "cell index is out of bounds" << std::endl;
    return;
  }
  auto &particle = particles_[cellidx];
  const auto &geo = particle.getGeometry();
  double max_dist = particle.getMaxInteractingDistance();
  double l0 = particle.getRestingEdgeLength();
  if (l0 < 1e-8) {
    std::cerr << "[DPM][Warning] Ideal Spring length (l0) must be greater than "
                 "0. Skipping surface interaction update."
              << std::endl;
    return;
  }

  // Build spatial grid only with boundary faces
  SpatialHashGrid face_grid(max_dist);
  for (size_t fi = 0; fi < faces.size(); fi++) {
    if (faces[fi].tet_b == -1) { // Only boundary faces
      face_grid.insert(0, fi, faces[fi].center);
    }
  }

  std::vector<SpatialHashGrid::CellVertex> nearby_faces;
  nearby_faces.reserve(50);

  for (size_t cell_face_idx = 0; cell_face_idx < geo.nFaces();
       cell_face_idx++) {
    const auto &c_face = geo.getFaceIndices(cell_face_idx);
    int n_partners = 0;
    Eigen::Vector3d force = Eigen::Vector3d::Zero();
    for (size_t vi = 0; vi < geo.nVerts(); vi++) {
      auto &vert = particle.getMutVertexMeta(vi);
      vert.is_focal_adhesion = false;
    }
    const Eigen::Vector3d &cf_center = geo.getFaceCentroid(cell_face_idx);
    const Eigen::Vector3d &cNormal = geo.getFaceNormals(cell_face_idx);

    queryFaceNeighbors(cf_center, max_dist, face_grid, nearby_faces);

    for (const auto &neighbor : nearby_faces) {
      size_t fi = neighbor.vertex_idx;
      const Eigen::Vector3d &mf_center = faces[fi].center;
      Eigen::Vector3d rij = mf_center - cf_center;
      double dist2 = rij.squaredNorm();
      if (dist2 > max_dist * max_dist)
        continue;
      const Eigen::Vector3d &mNormal = -faces[fi].normal;
      bool facesEachOther = cNormal.dot(mNormal) < 0;
      if (!facesEachOther)
        continue;
      n_partners++;
      Eigen::Vector3d ftmp = ((std::sqrt(dist2) / l0) - 1.0) *
                             (geo.getCentroid() - mf_center).normalized();
      force += DeformableParticle::Ks * ftmp;
    }
    if (n_partners == 0)
      continue;
    force /= static_cast<double>(n_partners);
    for (int vi = 0; vi < 3; vi++) {
      particle.addSurfaceAdhesionForce(c_face[vi], force);
      particle.getMutVertexMeta(c_face[vi]).is_focal_adhesion = true;
    }
  }
}

void ParticleInteractions::closestNeighborUpdate() {
  for (size_t ci = 0; ci < particles_.size(); ci++) {
    closestNeighborUpdate(ci);
  }
}

void ParticleInteractions::closestNeighborUpdate(const size_t ci) {
  for (size_t vi = 0; vi < particles_[ci].getGeometry().nVerts(); vi++) {
    closestNeighborUpdate(ci, vi);
  }
}

void ParticleInteractions::closestNeighborUpdate(const size_t ci,
                                                 const size_t vi) {
  double min_dist = DBL_MAX;
  int closestCellIdx = -1;
  int closestVertIdx = -1;

  const Eigen::Vector3d &pi = particles_[ci].getGeometry().getPosition(vi);

  double search_radius = particles_[ci].getMaxInteractingDistance() * 3.0;
  std::vector<SpatialHashGrid::CellVertex> neighbors;
  neighbors.reserve(100);

  while (closestCellIdx == -1 && search_radius < 100.0) {
    queryNeighbors(pi, search_radius, neighbors);

    for (const auto &neighbor : neighbors) {
      if (neighbor.cell_idx == static_cast<int>(ci))
        continue;

      double dist = (pi - neighbor.position).norm();
      if (dist < min_dist) {
        min_dist = dist;
        closestCellIdx = neighbor.cell_idx;
        closestVertIdx = neighbor.vertex_idx;
      }
    }

    if (closestCellIdx == -1) {
      search_radius *= 2.0;
    }
  }

  particles_[ci].getMutVertexMeta(vi).closest_cell_index = closestCellIdx;
  particles_[ci].getMutVertexMeta(vi).closest_vert_index = closestVertIdx;
}

std::vector<double>
ParticleInteractions::findWindingNumbersBetween(DeformableParticle &p1,
                                                DeformableParticle &p2) {
  std::vector<double> windingNumbers(p1.getGeometry().nVerts(), 0.0);
  Eigen::Vector3d minA = Eigen::Vector3d::Constant(DBL_MAX);
  Eigen::Vector3d maxA = Eigen::Vector3d::Constant(-DBL_MAX);
  Eigen::Vector3d minB = Eigen::Vector3d::Constant(DBL_MAX);
  Eigen::Vector3d maxB = Eigen::Vector3d::Constant(-DBL_MAX);

  for (size_t i = 0; i < p1.getGeometry().nVerts(); i++) {
    const auto &v = p1.getGeometry().getPosition(i);
    minA = minA.cwiseMin(v);
    maxA = maxA.cwiseMax(v);
  }
  for (size_t i = 0; i < p2.getGeometry().nVerts(); i++) {
    const auto &v = p2.getGeometry().getPosition(i);
    minB = minB.cwiseMin(v);
    maxB = maxB.cwiseMax(v);
  }

  bool overlapX = maxA.x() >= minB.x() && minA.x() <= maxB.x();
  bool overlapY = maxA.y() >= minB.y() && minA.y() <= maxB.y();
  bool overlapZ = maxA.z() >= minB.z() && minA.z() <= maxB.z();
  bool BBoverlap = overlapX && overlapY && overlapZ;
  if (!BBoverlap)
    return windingNumbers;

  for (size_t vi = 0; vi < p1.getGeometry().nVerts(); vi++) {
    const Eigen::Vector3d &point = p1.getGeometry().getPosition(vi);
    windingNumbers[vi] = p2.getGeometry().getWindingNumber(point);
  }
  return windingNumbers;
}
