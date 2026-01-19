#pragma once

#include "Polyhedron/Polyhedron.hpp"
#include <vector>

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
  std::vector<Eigen::Vector3d> Fv_;  // Volume forces
  std::vector<Eigen::Vector3d> Fa_;  // Area Forces
  std::vector<Eigen::Vector3d> Fb_;  // Bending Forces
  std::vector<Eigen::Vector3d> Fs_;  // Surface/ECM Adhesion Forces
  std::vector<Eigen::Vector3d> Fat_; // Cell-Cell junction/attraction forces
  std::vector<Eigen::Vector3d> Fre_; // Overlap/repulsion forces

  // Cell-Fluid Forces
  std::vector<Eigen::Vector3d>
      shear_stress_; // shear stress intep from stokes fem
  std::vector<Eigen::Vector3d>
      pressure_forces_; // pressure force interp from stokes

  // total force vector
  std::vector<Eigen::Vector3d> sum_forces_;

  // vertex meta data
  std::vector<VertMeta> vertex_meta_;

  Polyhedron shape_;

public:
  // Constructors
  explicit DeformableParticle(const Eigen::Vector3d &starting_point,
                              const double shape_param, const int f,
                              const double r0, const double Kv, const double Ka,
                              const double Kb);
  explicit DeformableParticle(const int f, const double radius)
      : DeformableParticle(Eigen::Vector3d{0, 0, 0}, 1.0, f, radius, 0.0, 0.0,
                           0.0) {
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
  const std::vector<Eigen::Vector3d> &getVolumeForces() const { return Fv_; }
  const std::vector<Eigen::Vector3d> &getAreaForces() const { return Fa_; }
  const std::vector<Eigen::Vector3d> &getBendingForces() const { return Fb_; }
  const std::vector<Eigen::Vector3d> &getMatrixAdhesionForces() const {
    return Fs_;
  }
  const std::vector<Eigen::Vector3d> &getCellAdhesionForces() const {
    return Fat_;
  }
  const std::vector<Eigen::Vector3d> &getCellRepulsiveForces() const {
    return Fre_;
  }
  const std::vector<Eigen::Vector3d> &getShearForces() const {
    return shear_stress_;
  }
  const std::vector<Eigen::Vector3d> &getPressureForces() const {
    return pressure_forces_;
  }
  const std::vector<Eigen::Vector3d> &getTotalForces() const {
    return sum_forces_;
  }
  double getMaxInteractingDistance() const { return max_dist_; }
  double getRestingEdgeLength() const { return l0_; }
  const VertMeta &getVertexMetaData(const size_t index) const {
    assert(index < vertex_meta_.size());
    return vertex_meta_[index];
  }
  VertMeta &getMutVertexMeta(const size_t index) {
    assert(index < vertex_meta_.size());
    return vertex_meta_[index];
  }

  // Setters
  void setAttactionForce(const size_t force_index,
                         const Eigen::Vector3d &force) {
    assert(force_index < Fat_.size());
    Fat_[force_index] = force;
  }
  void addAttactionForce(const size_t force_index,
                         const Eigen::Vector3d &force) {
    assert(force_index < Fat_.size());
    Fat_[force_index] += force;
  }
  void setRepulsiveForce(const size_t force_index,
                         const Eigen::Vector3d &force) {
    assert(force_index < Fat_.size());
    Fre_[force_index] = force;
  }
  void addRepulsiveForce(const size_t force_index,
                         const Eigen::Vector3d &force) {
    assert(force_index < Fat_.size());
    Fre_[force_index] += force;
  }
  void setSurfaceAdhesionForce(const size_t force_index,
                               Eigen::Vector3d &force) {
    assert(force_index < Fs_.size());
    Fs_[force_index] = force;
  }
  void addSurfaceAdhesionForce(const size_t force_index,
                               Eigen::Vector3d &force) {
    assert(force_index < Fs_.size());
    Fs_[force_index] += force;
  }

  // Updaters :)
  void volumeForceUpdate();
  void surfaceAreaForceUpdate();
  void bendingForceUpdate();
  void ShapeForcesUpdate();
  void moveTo(const Eigen::Vector3d &position);
  void resetForces();
  void mergeForces();
  void eulerUpdatePositions(double dt);
};
