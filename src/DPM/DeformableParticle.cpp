#include "DPM/DeformableParticle.hpp"
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

void DeformableParticle::bendingForceUpdate() {
  if (Kb_ < 1e-12) {
    return;
  }
  const auto &adj = shape_.getAdjaceny();
  for (size_t vi = 0; vi < shape_.nVerts(); vi++) {
    auto it = adj.find(vi);
    if (it == adj.end()) {
      return;
    }
    const auto &neighbors = it->second;

    if (neighbors.size() == 0) {
      continue;
    }
    glm::dvec3 avg{0.0};
    for (const int &n : neighbors) {
      avg += shape_.getPosition(n);
    }
    avg /= static_cast<double>(shape_.nVerts());
    Fb_[vi] += Kb_ * (avg - shape_.getPosition(vi));
  }
}

void DeformableParticle::volumeForceUpdate() {
  if (Kv_ < 1e-12) {
    return;
  }
  double volume = shape_.getVolume();
  double volumeStrain = (volume / v0_) - 1.0;
  for (size_t fi = 0; fi < shape_.nFaces(); fi++) {
    const auto &face = shape_.getFaceIndices(fi);
    glm::dvec3 a = shape_.getPosition(face[1]) - shape_.getPosition(face[0]);
    glm::dvec3 b = shape_.getPosition(face[2]) - shape_.getPosition(face[0]);
    glm::dvec3 c = glm::normalize(glm::cross(a, b));
    glm::dvec3 force = c * -Kv_ * volumeStrain * 0.3;
    Fv_[face[0]] += force;
    Fv_[face[1]] += force;
    Fv_[face[2]] += force;
  }
}

void DeformableParticle::surfaceAreaForceUpdate() {
  if (Ka_ < 1e-12)
    return;
  double a0 = 4.0 * M_PI * std::powf(r0_, 2.0);
  double tmpl0 = std::sqrt((4.0 * a0) / std::sqrt(3.0));
  for (size_t fi = 0; fi < shape_.nFaces(); fi++) {
    const auto &f = shape_.getFaceIndices(fi);
    const glm::dvec3 &pos0 = shape_.getPosition(f[0]);
    const glm::dvec3 &pos1 = shape_.getPosition(f[1]);
    const glm::dvec3 &pos2 = shape_.getPosition(f[2]);
    glm::dvec3 lv0 = pos1 - pos0;
    glm::dvec3 lv1 = pos2 - pos1;
    glm::dvec3 lv2 = pos0 - pos2;
    glm::dvec3 lengths = {distance(pos1, pos0), distance(pos2, pos1),
                          distance(pos0, pos2)};
    glm::dvec3 ulv0 = lv0 / lengths[0];
    glm::dvec3 ulv1 = lv1 / lengths[1];
    glm::dvec3 ulv2 = lv2 / lengths[2];
    glm::dvec3 dli = {lengths[0] / tmpl0, lengths[1] / tmpl0,
                      lengths[2] / tmpl0};
    dli -= 1.0;
    Fa_[f[0]] += Ka_ * sqrt(a0) / tmpl0 * (dli[0] * ulv0 - dli[2] * ulv2);
    Fa_[f[1]] += Ka_ * sqrt(a0) / tmpl0 * (dli[1] * ulv1 - dli[0] * ulv0);
    Fa_[f[2]] += Ka_ * sqrt(a0) / tmpl0 * (dli[2] * ulv2 - dli[1] * ulv1);
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

void DeformableParticle::moveTo(const glm::dvec3 &position) {
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
    if (glm::any(glm::isnan(sum_forces_[vi]))) {
      throw std::runtime_error("[DPM] Error: NaN value found\n");
    }
    glm::dvec3 &p = shape_.getMutPosition(vi);
    p += dt * sum_forces_[vi];
  }
  shape_.updateGeometry();
}
