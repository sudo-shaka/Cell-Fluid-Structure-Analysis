#pragma once

#include <array>
#include <glm/vec3.hpp>
#include <map>
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
  void generateIsosphere(double r0, int n_recursions);
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
  const glm::dvec3 &get_position(size_t index) const {
    assert(index < positions_.size());
    return positions_[index];
  }
  glm::dvec3 &get_mut_position(size_t index) {
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

struct VertMeta {
  bool is_junction;
  bool is_focal_adhesion;
  int closest_cell_index;
  int closest_vert_index;
  double binding_prob;
  double new_binding_prob;
  double max_binding_prob;
  double ideal_force = 0.0;

  inline static double slip_param = 2.0;
  inline static double catch_param = 2.0;
  inline static double base_lifetime = 1.0;
};

class DeformableParticle {

  // initial params
  double calA0_; // ideal shape param
  double v0_;    // ideal volume
  double r0_;    // initial radius
  double l0_;    // resting edge length

  // stiffness constants
  double Kv_; // volume
  double Kb_; // bending
  double Ka_; // surface area

  double max_dist_; // maximum interacting distance

  // Cell/Tissue Forces
  std::vector<glm::dvec3> Fv_;  // Volume forces
  std::vector<glm::dvec3> Fa_;  // Area Forces
  std::vector<glm::dvec3> Fb_;  // Bending Forces
  std::vector<glm::dvec3> Fs_;  // Surface/ECM Adhesion Forces
  std::vector<glm::dvec3> Fat_; // Cell-Cell junction/attraction forces
  std::vector<glm::dvec3> Fre_; // Overlap/repulsion forces

  // Cell-Fluid Forces
  std::vector<glm::dvec3> shear_stress_; // shear stress intep from stokes fem
  std::vector<glm::dvec3> pressure_forces_; // pressure force interp from stokes

  // total force vector
  std::vector<glm::dvec3> sum_forces_;

  // vertex meta data
  std::vector<VertMeta> vertex_meta_;

  Polyhedron shape_;

public:
  // Constructors
  explicit DeformableParticle(const glm::dvec3 &starting_point,
                              const double shape_param, const int f,
                              const double r0, const double Kv, const double Ka,
                              const double Kb);
  explicit DeformableParticle(const int f, const double radius)
      : DeformableParticle(glm::dvec3{0}, 1.0, f, radius, 0.0, 0.0, 0.0) {
    /* Defaults to shape parameter of 1, at point {0,0,0} */
  }
  explicit DeformableParticle() : DeformableParticle(2, 1.0) {
    /* Defaults to recursion of 2, radius of 1.0 */
  }

  // interacting stiffnesses
  inline static double Ks = 0.0;  // Cell-Matrix/Mesh adhesion (ECM adhesion)
  inline static double Kat = 0.0; // Cell-Cell adhesion (Junctions)
  inline static double Kre = 0.0; // Cell-Cell repulsuin/overlap prevention

  // Getters
  const Polyhedron &getGeometry() const { return shape_; }
  const std::vector<glm::dvec3> &getVolumeForces() const { return Fv_; }
  const std::vector<glm::dvec3> &getAreaForces() const { return Fa_; }
  const std::vector<glm::dvec3> &getBendingForces() const { return Fb_; }
  const std::vector<glm::dvec3> &getMatrixAdhesionForces() const { return Fs_; }
  const std::vector<glm::dvec3> &getCellAdhesionForces() const { return Fat_; }
  const std::vector<glm::dvec3> &getCellRepulsiveForces() const { return Fre_; }
  const std::vector<glm::dvec3> &getShearForces() const {
    return shear_stress_;
  }
  const std::vector<glm::dvec3> &getPressureForces() const {
    return pressure_forces_;
  }
  const std::vector<glm::dvec3> &getTotalForces() const { return sum_forces_; }

  // Setters

  // Updaters :)
  void volumeForceUpdate();
  void surfaceAreaForceUpdate();
  void bendingForceUpdate();
  void ShapeForcesUpdate();
  void resetForces();
  bool eulerUpdatePositions(double dt) { return true; }
};
