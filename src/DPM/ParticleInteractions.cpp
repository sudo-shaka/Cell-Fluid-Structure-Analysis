#include "DPM/ParticleInteractions.hpp"
#include "Mesh/Mesh.hpp"

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
    /*grid.insert(0, fi, faces[fi].centroid); */
    // TODO:
  }
}
void ParticleInteractions::queryNeighbors(
    const glm::dvec3 &pos, double radius,
    std::vector<SpatialHashGrid::CellVertex> &out) const {
  spatial_grid_.queryNeighbors(pos, radius, out);
}
void ParticleInteractions::queryFaceNeighbors(
    const glm::dvec3 &pos, double radius, const SpatialHashGrid &grid,
    std::vector<SpatialHashGrid::CellVertex> &out) const {
  grid.queryNeighbors(pos, radius, out);
}

// interaction functions
void ParticleInteractions::disperseCellsToFaceCenters(
    const std::vector<Face> &faces) {}
void ParticleInteractions::interactingForceUpdate() {}
void ParticleInteractions::cellCellRepulsionUpdate(
    const size_t particle_index) {}
void ParticleInteractions::cellCellAttractionUpdate(
    const size_t particle_index) {}
void ParticleInteractions::cellMeshInteractionUpdate(
    const size_t particle_index) {}
