#include "DPM/ParticleInteractions.hpp"
#include "Mesh/Mesh.hpp"
#include <algorithm>
#include <random>

void ParticleInteractions::rebuildIntercellularSpatialGrid() {
  spatial_grid_.clear();
  for (size_t ci = 0; ci < particles_.size(); ci++) {
    const auto shape = particles_[ci].getGeometry();
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
  size_t n_faces = faces.size();

  std::vector<double> cum_prob(n_faces);
  double sum = 0.0;
  double total_area =
      std::accumulate(faces.begin(), faces.end(), 0.0,
                      [](double sum, const Face &s) { return sum + s.area; });
  for (size_t i = 0; i < n_faces; i++) {
    sum += faces[i].area / total_area;
    cum_prob[i] = sum;
  }
  for (size_t i = 0; i < particles_.size(); i++) {
    double rand = dist(gen);
    auto it = std::lower_bound(cum_prob.begin(), cum_prob.end(), rand);
    int face_index = std::distance(cum_prob.begin(), it);
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
    cellCellAttractionUpdate(pi);
  }
}
void ParticleInteractions::cellCellRepulsionUpdate(
    const size_t particle_index) {

  if (particles_[particle_index].Kre < 1e-12) {
    return;
  }
  const auto &particle = particles_[particle_index];
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
    for (size_t vi = 0; vi < geo.nVerts(); vi++) {
      const Eigen::Vector3d &position = geo.getPosition(vi);
      double winding_number =
          particles_[pj].getGeometry().getWindingNumber(position);
      if (winding_number < 1e-12) {
        continue;
      }
      Eigen::Vector3d force = (com - position).normalized();
      force *= DeformableParticle::Kre * winding_number;
      particles_[particle_index].addRepulsiveForce(vi, force);
    }
  }
}
void ParticleInteractions::cellCellAttractionUpdate(
    const size_t particle_index) {

  std::vector<SpatialHashGrid::CellVertex> neighbors;
  neighbors.reserve(100);

  auto &particle = particles_[particle_index];
  double max_dist = particle.getMaxInteractingDistance();
  double l0 = particle.getRestingEdgeLength();
  assert(l0 >= 1e-12);

  for (size_t vi = 0; vi < particle.getGeometry().nVerts(); vi++) {
    int n_parners = 0;
    Eigen::Vector3d force = Eigen::Vector3d::Zero();
    auto &vert_meta = particle.getMutVertexMeta(vi);
    vert_meta.is_junction = false;
    const auto &vert_position = particle.getGeometry().getPosition(vi);

    queryNeighbors(vert_position, max_dist, neighbors);

    for (const auto &neighbor : neighbors) {
      if (neighbor.cell_idx == static_cast<int>(particle_index))
        continue;
      double dist_sq = (neighbor.position - vert_position).squaredNorm();
      if (dist_sq >= max_dist * max_dist)
        continue;
      ++n_parners;
      force += DeformableParticle::Kat * 0.5 *
               ((std::sqrt(dist_sq) / l0) - 1.0) *
               (neighbor.position - vert_position).normalized();
    }
    if (n_parners == 0)
      continue;
    vert_meta.is_junction = true;
    force /= static_cast<double>(n_parners);
    particle.setAttactionForce(vi, force);
  }
}
void ParticleInteractions::cellMeshInteractionUpdate(
    const std::vector<Face> &faces, const size_t particle_index) {
  if (DeformableParticle::Ks < 1e-12)
    return;
  assert(particle_index < particles_.size());
  auto &particle = particles_[particle_index];
  const auto &geo = particle.getGeometry();
  double max_dist = particle.getMaxInteractingDistance();
  double l0 = particle.getRestingEdgeLength();
  assert(l0 > 1e-12);

  SpatialHashGrid face_grid(max_dist);
  rebuildMatrixFacesSpatialGrid(faces, face_grid);
  std::vector<SpatialHashGrid::CellVertex> nearby_faces;
  nearby_faces.reserve(50);

  for (size_t vi = 0; vi < geo.nVerts(); vi++) {
    auto &vert_meta = particle.getMutVertexMeta(vi);
    vert_meta.is_focal_adhesion = false;
  }

  for (size_t pfaceidx = 0; pfaceidx < geo.nFaces(); pfaceidx++) {
    const auto &p_face = geo.getFaceIndices(pfaceidx);
    int n_parners = 0;
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
      bool faces_each_other = p_face_normal.dot(mesh_face_normal) < 0.0;
      if (!faces_each_other)
        continue;
      n_parners++;
      const Eigen::Vector3d &com = geo.getCentroid();
      Eigen::Vector3d ftmp = ((std::sqrt(dist2) / l0) - 1.0) *
                             (com - mesh_face_center).normalized();
      force += DeformableParticle::Ks * ftmp;
    }
    if (n_parners == 0)
      continue;
    force /= static_cast<double>(n_parners);
    for (int vi = 0; vi < 3; vi++) {
      particle.addSurfaceAdhesionForce(p_face[vi], force);
      auto &vert_meta = particle.getMutVertexMeta(p_face[vi]);
      vert_meta.is_focal_adhesion = true;
    }
  }
}
