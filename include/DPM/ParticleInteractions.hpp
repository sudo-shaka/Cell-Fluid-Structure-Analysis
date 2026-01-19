#pragma once

#include "DPM/DeformableParticle.hpp"
#include <Eigen/Dense>
#include <stdexcept>
#include <unordered_map>
#include <vector>

struct SpatialHashGrid {
  struct GridKey {
    int x, y, z;
    bool operator==(const GridKey &other) const {
      return x == other.x && y == other.y && z == other.z;
    }
  };
  struct GridKeyHash {
    size_t operator()(const GridKey &k) const {
      return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1) ^
             (std::hash<int>()(k.z) << 2);
    }
  };

  struct CellVertex {
    int cell_idx;
    int vertex_idx;
    Eigen::Vector3d position;
  };

  double cell_size;
  std::unordered_map<GridKey, std::vector<CellVertex>, GridKeyHash> grid;

  explicit SpatialHashGrid(double size = 1.0) : cell_size(size) {}

  GridKey get_key(const Eigen::Vector3d &pos) const {
    return {static_cast<int>(std::floor(pos.x() / cell_size)),
            static_cast<int>(std::floor(pos.y() / cell_size)),
            static_cast<int>(std::floor(pos.z() / cell_size))};
  }

  void clear() { grid.clear(); }

  void insert(int cell_idx, int vertex_idx, const Eigen::Vector3d &pos) {
    GridKey key = get_key(pos);
    grid[key].push_back({cell_idx, vertex_idx, pos});
  }

  void queryNeighbors(const Eigen::Vector3d &pos, double radius,
                      std::vector<CellVertex> &neighbors) const {
    neighbors.clear();
    int range = static_cast<int>(std::ceil(radius / cell_size));
    GridKey center = get_key(pos);

    // Pre-allocate approximate capacity
    int expected_cells = (2 * range + 1) * (2 * range + 1) * (2 * range + 1);
    neighbors.reserve(expected_cells * 10); // Rough estimate

    for (int dx = -range; dx <= range; dx++) {
      for (int dy = -range; dy <= range; dy++) {
        for (int dz = -range; dz <= range; dz++) {
          GridKey key = {center.x + dx, center.y + dy, center.z + dz};
          auto it = grid.find(key);
          if (it != grid.end()) {
            for (const auto &cv : it->second) {
              double dist_sq = (pos - cv.position).squaredNorm();
              if (dist_sq <= radius * radius) {
                neighbors.push_back(cv);
              }
            }
          }
        }
      }
    }
  }
};

// forward declaration
struct Face;

class ParticleInteractions {

  std::vector<DeformableParticle> particles_; // particles that interact

  // Spatial grids to prevent O(N^2) interactor lookups
  SpatialHashGrid spatial_grid_;
  SpatialHashGrid ecm_spatial_grid_;
  void rebuildIntercellularSpatialGrid();
  void rebuildMatrixFacesSpatialGrid(const std::vector<Face> &faces,
                                     SpatialHashGrid &grid);
  void queryNeighbors(const Eigen::Vector3d &pos, double radius,
                      std::vector<SpatialHashGrid::CellVertex> &out) const;
  void queryFaceNeighbors(const Eigen::Vector3d &pos, double radius,
                          const SpatialHashGrid &grid,
                          std::vector<SpatialHashGrid::CellVertex> &out) const;
  // interaction functions
  void disperseCellsToFaceCenters(const std::vector<Face> &faces);
  void cellCellRepulsionUpdate(const size_t particle_index);
  void cellCellAttractionUpdate(const size_t particle_index);
  void cellMeshInteractionUpdate(const std::vector<Face> &mesh_faces,
                                 const size_t particle_index);

public:
  // Constructors
  explicit ParticleInteractions() = default;
  explicit ParticleInteractions(std::vector<DeformableParticle> particles)
      : particles_(particles) {
          /*TODO: update constuctor to handle spatial grid init*/
        };

  double nParticles() const { return particles_.size(); }

  // updates
  void interactingForceUpdate(const size_t particle_index);

  // getters
  const DeformableParticle &getParticle(size_t particle_index) const {
    if (particle_index > particles_.size()) {
      throw std::runtime_error("[ParticleInteraction] Trying to index a "
                               "particle outside of vector length\n");
    }
    return particles_[particle_index];
  }
};
