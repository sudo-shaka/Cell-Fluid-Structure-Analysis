#include "DPM/DeformableParticle.hpp"
#include <Eigen/StdVector>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

DeformableParticle::DeformableParticle(const Eigen::Vector3d &starting_point,
                                       const double shape_param, const int f,
                                       const double r0, const double Kv,
                                       const double Ka, const double Kb)
    : calA0_(shape_param), r0_(r0), Kv_(Kv), Kb_(Kb), Ka_(Ka),
      shape_(Polyhedron::isosphere(r0, f)) {

  // Calculate number of faces and vertices based on subdivision level f
  int n_faces = 20 * int(std::pow(4, double(f)));

  // Calculate ideal volume and surface area
  v0_ = (4.0 / 3.0) * M_PI * std::pow(r0, 3.0);
  sa0_ = std::pow((6.0 * std::sqrt(M_PI) * v0_ * shape_param), (2.0 / 3.0));
  a0_ = sa0_ / (double)n_faces;
  l0_ = r0 / 3.0;

  // Set maximum interaction distance
  max_dist_ = r0 / 3.0;

  // Move shape to starting position
  shape_.moveTo(starting_point);

  // Initialize force vectors
  size_t nverts = shape_.nVerts();
  Fv_.resize(nverts, Eigen::Vector3d::Zero());
  Fa_.resize(nverts, Eigen::Vector3d::Zero());
  Fb_.resize(nverts, Eigen::Vector3d::Zero());
  Fs_.resize(nverts, Eigen::Vector3d::Zero());
  Fat_.resize(nverts, Eigen::Vector3d::Zero());
  Fre_.resize(nverts, Eigen::Vector3d::Zero());
  shear_stress_.resize(nverts, Eigen::Vector3d::Zero());
  pressure_forces_.resize(nverts, Eigen::Vector3d::Zero());
  sum_forces_.resize(nverts, Eigen::Vector3d::Zero());

  // Initialize vertex metadata
  vertex_meta_.resize(nverts);
  for (auto &meta : vertex_meta_) {
    meta.is_junction = false;
    meta.is_focal_adhesion = false;
    meta.closest_cell_index = -1;
    meta.closest_vert_index = -1;
    meta.binding_prob = 0.0;
    meta.new_binding_prob = 0.0;
    meta.max_binding_prob = 1.0;
    meta.ideal_force = 0.0;
  }
}

void DeformableParticle::bendingForceUpdate() {
  if (Kb_ < 1e-8) {
    return;
  }
  const auto &adj = shape_.getAdjaceny();
  for (size_t vi = 0; vi < shape_.nVerts(); vi++) {
    auto it = adj.find(vi);
    if (it == adj.end()) {
      continue;
    }
    const auto &neighbors = it->second;

    if (neighbors.size() == 0) {
      continue;
    }
    Eigen::Vector3d avg = Eigen::Vector3d::Zero();
    for (const int &n : neighbors) {
      avg += shape_.getPosition(n);
    }
    avg /= static_cast<double>(neighbors.size());
    Fb_[vi] += Kb_ * (avg - shape_.getPosition(vi));
  }
}

void DeformableParticle::volumeForceUpdate() {
  if (Kv_ < 1e-8) {
    return;
  }
  double volume = shape_.getVolume();
  double volumeStrain = (volume / v0_) - 1.0;
  for (size_t fi = 0; fi < shape_.nFaces(); fi++) {
    const auto &face = shape_.getFaceIndices(fi);
    Eigen::Vector3d A =
        shape_.getPosition(face[1]) - shape_.getPosition(face[0]);
    Eigen::Vector3d B =
        shape_.getPosition(face[2]) - shape_.getPosition(face[0]);
    Eigen::Vector3d cross = A.cross(B);
    double cross_norm = cross.norm();

    // Avoid division by zero when normalizing
    if (cross_norm < 1e-10)
      continue;

    Eigen::Vector3d C = cross / cross_norm;
    Eigen::Vector3d force = C * -Kv_ * volumeStrain * 0.3;
    Fv_[face[0]] += force;
    Fv_[face[1]] += force;
    Fv_[face[2]] += force;
  }
}

void DeformableParticle::surfaceAreaForceUpdate() {
  if (Ka_ < 1e-8)
    return;
  double tmpl0 = std::sqrt((4.0 * a0_) / std::sqrt(3.0));
  for (size_t fi = 0; fi < shape_.nFaces(); fi++) {
    const auto &f = shape_.getFaceIndices(fi);
    const Eigen::Vector3d &pos0 = shape_.getPosition(f[0]);
    const Eigen::Vector3d &pos1 = shape_.getPosition(f[1]);
    const Eigen::Vector3d &pos2 = shape_.getPosition(f[2]);
    Eigen::Vector3d lv0 = pos1 - pos0;
    Eigen::Vector3d lv1 = pos2 - pos1;
    Eigen::Vector3d lv2 = pos0 - pos2;
    Eigen::Vector3d lengths;
    lengths << lv0.norm(), lv1.norm(), lv2.norm();

    // Avoid division by zero
    if (lengths[0] < 1e-10 || lengths[1] < 1e-10 || lengths[2] < 1e-10)
      continue;

    Eigen::Vector3d ulv0 = lv0 / lengths[0];
    Eigen::Vector3d ulv1 = lv1 / lengths[1];
    Eigen::Vector3d ulv2 = lv2 / lengths[2];
    Eigen::Vector3d dli;
    dli << lengths[0] / tmpl0, lengths[1] / tmpl0, lengths[2] / tmpl0;
    dli.array() -= 1.0;
    Fa_[f[0]] += Ka_ * std::sqrt(a0_) / tmpl0 * (dli[0] * ulv0 - dli[2] * ulv2);
    Fa_[f[1]] += Ka_ * std::sqrt(a0_) / tmpl0 * (dli[1] * ulv1 - dli[0] * ulv0);
    Fa_[f[2]] += Ka_ * std::sqrt(a0_) / tmpl0 * (dli[2] * ulv2 - dli[1] * ulv1);
  }
}

void DeformableParticle::ShapeForcesUpdate() {
  surfaceAreaForceUpdate();
  bendingForceUpdate();
  volumeForceUpdate();
}

void DeformableParticle::resetForces() {
  for (size_t vi = 0; vi < shape_.nVerts(); vi++) {
    Fat_[vi] = {0, 0, 0};
    Fre_[vi] = {0, 0, 0};
    Fa_[vi] = {0, 0, 0};
    Fb_[vi] = {0, 0, 0};
    Fv_[vi] = {0, 0, 0};
    Fs_[vi] = {0, 0, 0};
  }
}

void DeformableParticle::moveTo(const Eigen::Vector3d &position) {
  shape_.moveTo(position);
}

void DeformableParticle::mergeForces() {
  size_t nverts = shape_.nVerts();
  assert(sum_forces_.size() == nverts);
  assert(Fat_.size() == nverts);
  assert(Fre_.size() == nverts);
  assert(Fv_.size() == nverts);
  assert(Fb_.size() == nverts);
  assert(Fa_.size() == nverts);
  for (size_t vi = 0; vi < nverts; vi++) {
    sum_forces_[vi] =
        Fat_[vi] + Fre_[vi] + Fv_[vi] + Fs_[vi] + Fb_[vi] + Fa_[vi];
  }
}

void DeformableParticle::eulerUpdatePositions(const double dt) {
  mergeForces();
  for (size_t vi = 0; vi < shape_.nVerts(); vi++) {
    if (sum_forces_[vi].array().isNaN().any()) {
      throw std::runtime_error("[DPM] Error: NaN value found\n");
    }
    Eigen::Vector3d &p = shape_.getMutPosition(vi);
    p += dt * sum_forces_[vi];
  }
  shape_.updateGeometry();
}
