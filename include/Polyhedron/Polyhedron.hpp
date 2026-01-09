#pragma once

#include <array>
#include <glm/vec3.hpp>
#include <map>
#include <string>
#include <vector>

class Polyhedron {
  double volume_;
  double surface_area_;
  glm::dvec3 centroid_;
  std::array<glm::dvec3, 2> bbox_min_max_;
  std::vector<double> face_areas_;
  std::vector<std::array<size_t, 3>> faces_;
  std::vector<glm::dvec3> face_normals_;
  std::vector<glm::dvec3> face_centers_;
  std::vector<glm::dvec3> positions_;
  std::map<std::pair<int, int>, int> mid_cache_;

  bool validate() const;
  double computeVolume() const;
  void updateFaceNormals();
  void updateFaceCenters();
  void updateFaceAreas();
  double computeFaceArea(size_t fi) const;
  void updateBoundingBox();
  void updateCentroid();
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
                      std::vector<glm::dvec3> input_positions);

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
  double getWindingNumber(const glm::dvec3 &point) const;
  bool pointInside(glm::dvec3 point) const {
    return getWindingNumber(point) > 0.5;
  }
  const glm::dvec3 &getCentroid() const { return centroid_; }
  const std::array<glm::dvec3, 2> &getBoundingBox() const {
    return bbox_min_max_;
  }
  const glm::dvec3 &getFaceCentroid(size_t face_index) const {
    return face_centers_[face_index];
  }
  double getVolume() const { return volume_; }
  double getSurfaceArea() const { return surface_area_; }
  const glm::dvec3 &getFaceNormals(size_t face_index) const {
    assert(face_index < face_normals_.size());
    return face_normals_[face_index];
  }
  const std::array<size_t, 3> &getFaceIndices(size_t fi) const {
    assert(fi < faces_.size());
    return faces_[fi];
  }
  const glm::dvec3 &getPosition(size_t index) const {
    assert(index < positions_.size());
    return positions_[index];
  }
  glm::dvec3 &getMutPosition(size_t index) {
    assert(index < positions_.size());
    return positions_[index];
  }
  size_t nFaces() const { return faces_.size(); }
  size_t nVerts() const { return positions_.size(); }
  // Setters
  void setPosition(size_t index, glm::dvec3 &position) {
    assert(index < positions_.size());
    positions_[index] = position;
  }
  // Updates
  void moveTo(const glm::dvec3 &point);
  void updateGeometry() {
    updateCentroid();
    updateFaceCenters();
    updateFaceNormals();
    updateFaceAreas();
    volume_ = computeVolume();
    updateBoundingBox();
  };
};
