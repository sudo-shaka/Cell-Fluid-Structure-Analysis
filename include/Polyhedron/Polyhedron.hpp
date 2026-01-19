#pragma once

#include <Eigen/Dense>
#include <array>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class Polyhedron {
  double volume_;
  double surface_area_;
  Eigen::Vector3d centroid_;
  std::pair<Eigen::Vector3d, Eigen::Vector3d> bbox_min_max_;
  std::vector<double> face_areas_;
  std::vector<std::array<size_t, 3>> faces_;
  std::vector<Eigen::Vector3d> face_normals_;
  std::vector<Eigen::Vector3d> face_centers_;
  std::vector<Eigen::Vector3d> positions_;
  std::map<std::pair<int, int>, int> mid_cache_;
  std::unordered_map<int, std::unordered_set<int>> adjacency_;

  bool validate() const;
  double computeVolume() const;
  void updateFaceNormals();
  void updateFaceCenters();
  void updateFaceAreas();
  double computeFaceArea(size_t fi) const;
  void updateBoundingBox();
  void updateCentroid();
  void buildAdjecency();
  void generateIsosphere(const double r0, const int n_recursions);
  void generateCylendar(const double length, const double radius,
                        const int resolution);
  void generateFromObjFile(const std::string &filename);
  int getOrCreateMidpoint(int v1, int v2,
                          std::map<std::pair<int, int>, int> *cache);

public:
  // Constructors
  explicit Polyhedron() = default;
  explicit Polyhedron(std::vector<std::array<size_t, 3>> input_faces,
                      std::vector<Eigen::Vector3d> input_positions);

  // statics
  static Polyhedron isosphere(double radius, int recursion_level = 2) {
    Polyhedron p;
    p.generateIsosphere(radius, recursion_level);
    return p;
  };
  static Polyhedron fromWavefront(const std::string &filename) {
    Polyhedron p;
    p.generateFromObjFile(filename);
    return p;
  };
  static Polyhedron cylendar(const double length, const double radius,
                             const int resolution) {
    Polyhedron p;
    p.generateCylendar(length, radius, resolution);
    return p;
  }

  // Getters
  double getWindingNumber(const Eigen::Vector3d &point) const;
  bool pointInside(Eigen::Vector3d point) const {
    return getWindingNumber(point) > 0.5;
  }
  const Eigen::Vector3d &getCentroid() const { return centroid_; }
  const std::pair<Eigen::Vector3d, Eigen::Vector3d> &getBoundingBox() const {
    return bbox_min_max_;
  }
  const Eigen::Vector3d &getFaceCentroid(size_t face_index) const {
    return face_centers_[face_index];
  }
  double getVolume() const { return volume_; }
  double getSurfaceArea() const { return surface_area_; }
  const Eigen::Vector3d &getFaceNormals(size_t face_index) const {
    assert(face_index < face_normals_.size());
    return face_normals_[face_index];
  }
  const std::array<size_t, 3> &getFaceIndices(size_t fi) const {
    assert(fi < faces_.size());
    return faces_[fi];
  }
  const Eigen::Vector3d &getPosition(size_t index) const {
    assert(index < positions_.size());
    return positions_[index];
  }
  Eigen::Vector3d &getMutPosition(size_t index) {
    assert(index < positions_.size());
    return positions_[index];
  }
  size_t nFaces() const { return faces_.size(); }
  size_t nVerts() const { return positions_.size(); }
  const std::unordered_map<int, std::unordered_set<int>> &getAdjaceny() const {
    return adjacency_;
  }
  // Setters
  void setPosition(size_t index, Eigen::Vector3d &position) {
    assert(index < positions_.size());
    positions_[index] = position;
  }
  // Updates
  void moveTo(const Eigen::Vector3d &point);
  void updateGeometry() {
    updateCentroid();
    updateFaceCenters();
    updateFaceNormals();
    updateFaceAreas();
    volume_ = computeVolume();
    updateBoundingBox();
  };
};
