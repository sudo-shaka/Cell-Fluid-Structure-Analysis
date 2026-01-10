#include "Polyhedron/Polyhedron.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <iostream>
#include <sstream>
#include <stdexcept>

Polyhedron::Polyhedron(std::vector<std::array<size_t, 3>> input_faces,
                       std::vector<glm::dvec3> input_positions)
    : faces_(input_faces), positions_(input_positions) {
  if (faces_.empty() || positions_.size() < 3) {
    std::cerr << "[Polyhedron] constructor vectors do not contain enough data "
                 "for polyhedron generation\n";
    return;
  }
  updateGeometry();
  bool okay = validate();
  if (!okay) {
    throw std::runtime_error("[Polyhedrion] validation failed during "
                             "construction. Check input vectors.\n");
  }
}

void Polyhedron::generateIsosphere(double radius, int recursion_level) {
  if (recursion_level >= 16) {
    std::cerr << "[Polyhedron]"
              << "cannot create polyhedron with more than 16 recursions\n";
  }
  const int pow4 = 1 << (2 * recursion_level); // 4 ^ f
  size_t n_faces = 20 * static_cast<size_t>(pow4);
  size_t n_verts = 10 * static_cast<size_t>(pow4) + 2;

  positions_.reserve(n_verts);
  faces_.reserve(n_faces);
  double t = (1.0 + std::sqrt(5.0)) / 2.0;
  positions_ = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
                {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
                {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
  for (auto &p : positions_) {
    p = glm::normalize(p);
  }

  // Define faces of the icosahedron (20 triangles)
  // 5 faces around point 0
  faces_.push_back(std::array<size_t, 3>{0, 11, 5});
  faces_.push_back(std::array<size_t, 3>{0, 5, 1});
  faces_.push_back(std::array<size_t, 3>{0, 1, 7});
  faces_.push_back(std::array<size_t, 3>{0, 7, 10});
  faces_.push_back(std::array<size_t, 3>{0, 10, 11});

  // 5 adjacent facess
  faces_.push_back(std::array<size_t, 3>{1, 5, 9});
  faces_.push_back(std::array<size_t, 3>{5, 11, 4});
  faces_.push_back(std::array<size_t, 3>{11, 10, 2});
  faces_.push_back(std::array<size_t, 3>{10, 7, 6});
  faces_.push_back(std::array<size_t, 3>{7, 1, 8});

  // 5 faces around point 3
  faces_.push_back(std::array<size_t, 3>{3, 9, 4});
  faces_.push_back(std::array<size_t, 3>{3, 4, 2});
  faces_.push_back(std::array<size_t, 3>{3, 2, 6});
  faces_.push_back(std::array<size_t, 3>{3, 6, 8});
  faces_.push_back(std::array<size_t, 3>{3, 8, 9});

  // 5 adjacent faces
  faces_.push_back(std::array<size_t, 3>{4, 9, 5});
  faces_.push_back(std::array<size_t, 3>{2, 4, 11});
  faces_.push_back(std::array<size_t, 3>{6, 2, 10});
  faces_.push_back(std::array<size_t, 3>{8, 6, 7});
  faces_.push_back(std::array<size_t, 3>{9, 8, 1});

  // Subdivide faces recursively to the requested level
  for (int i = 0; i < recursion_level; ++i) {
    std::vector<std::array<size_t, 3>> new_faces;
    std::map<std::pair<int, int>, int> cache;
    new_faces.reserve(faces_.size() * 4);

    for (const auto &f : faces_) {
      int a = static_cast<int>(f[0]);
      int b = static_cast<int>(f[1]);
      int c = static_cast<int>(f[2]);

      int ab = getOrCreateMidpoint(a, b, &cache);
      int bc = getOrCreateMidpoint(b, c, &cache);
      int ca = getOrCreateMidpoint(c, a, &cache);

      new_faces.push_back(std::array<size_t, 3>{static_cast<size_t>(a),
                                                static_cast<size_t>(ab),
                                                static_cast<size_t>(ca)});
      new_faces.push_back(std::array<size_t, 3>{static_cast<size_t>(b),
                                                static_cast<size_t>(bc),
                                                static_cast<size_t>(ab)});
      new_faces.push_back(std::array<size_t, 3>{static_cast<size_t>(c),
                                                static_cast<size_t>(ca),
                                                static_cast<size_t>(bc)});
      new_faces.push_back(std::array<size_t, 3>{static_cast<size_t>(ab),
                                                static_cast<size_t>(bc),
                                                static_cast<size_t>(ca)});
    }

    faces_.swap(new_faces);
  }
  // Project all vertices to the requested radius
  for (auto &p : positions_) {
    p = glm::normalize(p) * radius;
  }

  updateGeometry();
  assert(validate());
}

void Polyhedron::generateCylendar(const double length, const double radius,
                                  const int resolution) {
  size_t num_circum_div = std::max(3, resolution);
  size_t num_axial_div = std::max(2, resolution);
  faces_.clear();
  positions_.clear();
  // create all axial rows
  for (size_t j = 0; j < num_axial_div; j++) {
    double dj = static_cast<double>(j);
    double x = (num_axial_div == 1)
                   ? 0.0
                   : (dj * length / static_cast<double>(num_axial_div - 1));
    for (size_t i = 0; i < num_circum_div; i++) {
      double di = static_cast<double>(i);
      double theta = 2.0 * M_PI * di / static_cast<double>(num_circum_div);
      double y = radius * std::cos(theta);
      double z = radius * std::sin(theta);
      positions_.push_back(glm::dvec3{x, y, z});
    }
  }

  // iterate only over axial rows that have a row below them
  for (size_t j = 0; j + 1 < num_axial_div; j++) {
    for (size_t i = 0; i < num_circum_div; i++) {
      size_t idx = j * num_circum_div + i;
      size_t right = j * num_circum_div + ((i + 1) % num_circum_div);
      size_t down = (j + 1) * num_circum_div + i;
      size_t downRight = (j + 1) * num_circum_div + ((i + 1) % num_circum_div);
      // two triangles per quad: choose consistent winding; will fix globally
      // later
      faces_.push_back({idx, down, right});       // triangle 1
      faces_.push_back({right, down, downRight}); // triangle 2
    }
  }

  // bottom cap (x = 0) - fan triangulation
  glm::dvec3 bottom_center = glm::dvec3{0.0, 0.0, 0.0};
  size_t bottom_center_index = positions_.size();
  positions_.push_back(bottom_center);
  for (size_t i = 0; i < num_circum_div; ++i) {
    size_t next = (i + 1) % num_circum_div;
    // add triangle (center, next, i) - orientation fixed later
    faces_.push_back({bottom_center_index, next, i});
  }

  // top cap (x = length)
  glm::dvec3 top_center = glm::dvec3{length, 0.0, 0.0};
  size_t top_center_idx = positions_.size();
  positions_.push_back(top_center);
  int start_top_row = (num_axial_div - 1) * num_circum_div;
  for (size_t i = 0; i < num_circum_div; ++i) {
    size_t curr = start_top_row + i;
    size_t next = start_top_row + ((i + 1) % num_circum_div);
    // add triangle (center, curr, next) - orientation fixed later
    faces_.push_back({top_center_idx, curr, next});
  }
  updateGeometry();
  assert(validate());
}

void Polyhedron::generateFromObjFile(const std::string &filename) {
  std::filesystem::path p(filename);
  if (!p.has_extension() || p.extension() != ".obj") {
    std::cerr << "[Polyhedrion] Warning. Wavefront (.obj) file extension not "
                 "used. Double check this is correct."
              << "\n[Polyhedron] Continuing to try to parse entered file."
              << std::endl;
  }
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }
  faces_.clear();
  positions_.clear();
  std::string line;
  std::string currentgroup;
  std::vector<std::array<size_t, 3>> faces;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string word;
    ss >> word;
    if (word == "v") {
      double x, y, z;
      ss >> x >> y >> z;
      positions_.push_back({x, y, z});
    } else if (word == "f") {
      size_t f1, f2, f3;
      ss >> f1 >> f2 >> f3;
      std::array<size_t, 3> face = {f1 - 1, f2 - 1, f3 - 1};
      faces_.push_back(face);
    }
  }
  file.close();
  updateGeometry();
  assert(validate());
}

double Polyhedron::computeVolume() const {
  double volume = 0.0;
  for (const auto &face : faces_) {
    const glm::dvec3 &p0 = positions_[face[0]];
    const glm::dvec3 &p1 = positions_[face[1]];
    const glm::dvec3 &p2 = positions_[face[2]];
    const glm::dvec3 cross = glm::cross(p1, p2);
    volume += glm::dot(p0, cross);
  }
  return std::abs(volume) / 6.0;
}

double Polyhedron::getWindingNumber(const glm::dvec3 &point) const {
  double totalOmega = 0.0;
  for (const auto &f : faces_) {
    const glm::dvec3 &a = positions_[f[0]] - point;
    const glm::dvec3 &b = positions_[f[1]] - point;
    const glm::dvec3 &c = positions_[f[2]] - point;

    if (glm::length(a) < 1e-8 || glm::length(b) < 1e-8 || glm::length(c) < 1e-8)
      continue;

    glm::dvec3 u = glm::normalize(a);
    glm::dvec3 v = glm::normalize(b);
    glm::dvec3 w = glm::normalize(c);

    double denom = 1.0 + glm::dot(u, v) + glm::dot(v, w) + glm::dot(w, u);
    if (denom < 1e-8)
      continue;

    double num = glm::dot(u, glm::cross(v, w));
    double omega = 2.0 * std::atan2(num, denom);

    if (std::fabs(omega) > 1e-10) {
      totalOmega += omega;
    }
  }
  return totalOmega / (4.0 * M_PI);
}

void Polyhedron::updateCentroid() {
  centroid_ = glm::dvec3{0.0};
  for (const auto p : positions_) {
    centroid_ += p;
  }
  centroid_ /= static_cast<double>(positions_.size());
}

void Polyhedron::updateFaceNormals() {
  if (face_normals_.size() != faces_.size()) {
    face_normals_.resize(faces_.size());
  }
  for (size_t fi = 0; fi < faces_.size(); fi++) {
    const auto &f = faces_[fi];
    const glm::dvec3 &p0 = positions_[f[0]];
    const glm::dvec3 &p1 = positions_[f[1]];
    const glm::dvec3 &p2 = positions_[f[2]];
    face_normals_[fi] = glm::normalize(glm::cross(p1 - p0, p2 - p0));
  }
}

void Polyhedron::updateFaceCenters() {
  const size_t n_faces = faces_.size();
  if (face_centers_.size() != n_faces) {
    face_centers_.resize(n_faces);
  }
  for (size_t fi = 0; fi < n_faces; fi++) {
    const auto &f = faces_[fi];
    const glm::dvec3 &p0 = positions_[f[0]];
    const glm::dvec3 &p1 = positions_[f[1]];
    const glm::dvec3 &p2 = positions_[f[2]];
    face_centers_[fi] = (p0 + p1 + p2) / 3.0;
  }
}

void Polyhedron::updateFaceAreas() {
  if (face_areas_.size() != faces_.size()) {
    face_areas_.resize(faces_.size());
  }
  surface_area_ = 0.0;
  for (size_t fi = 0; fi < faces_.size(); fi++) {
    double area = computeFaceArea(fi);
    face_areas_[fi] = area;
    surface_area_ += area;
  }
}

double Polyhedron::computeFaceArea(size_t fi) const {
  const glm::dvec3 &a = positions_[faces_[fi][0]];
  const glm::dvec3 &b = positions_[faces_[fi][1]];
  const glm::dvec3 &c = positions_[faces_[fi][2]];
  // Use cross-product based area: 0.5 * |(b-a) x (c-a)|
  glm::dvec3 ab = b - a;
  glm::dvec3 ac = c - a;
  double area = 0.5 * glm::length(glm::cross(ab, ac));
  return area;
}

void Polyhedron::updateBoundingBox() {
  assert(!positions_.empty());
  glm::dvec3 min = positions_[0];
  glm::dvec3 max = positions_[0];
  for (const auto &p : positions_) {
    min = glm::min(min, p);
    max = glm::max(max, p);
  }
  bbox_min_max_ = {min, max};
}

void Polyhedron::moveTo(const glm::dvec3 &point) {
  glm::dvec3 offset = point - centroid_;
  for (auto &vert : positions_) {
    vert += offset;
  }
  updateGeometry();
}

bool Polyhedron::validate() const {
  if (positions_.empty())
    return false;
  if (faces_.empty())
    return false;
  size_t n_verts = positions_.size();
  size_t n_faces = faces_.size();

  for (const auto &p : positions_) {
    if (glm::any(glm::isnan(p))) {
      return false;
    }
  }

  for (const auto &f : faces_) {
    if (f[0] >= n_verts || f[1] >= n_verts || f[2] >= n_verts) {
      return false;
    }
  }

  if (face_centers_.size() != n_faces)
    return false;
  if (face_normals_.size() != n_faces)
    return false;
  if (face_areas_.size() != n_faces)
    return false;
  if (!pointInside(centroid_))
    return false;

  return true;
}
int Polyhedron::getOrCreateMidpoint(int a, int b,
                                    std::map<std::pair<int, int>, int> *cache) {
  // canonical order
  auto key = std::minmax(a, b);

  auto it = cache->find(key);
  if (it != cache->end())
    return it->second;

  const glm::dvec3 &pa = positions_[a];
  const glm::dvec3 &pb = positions_[b];

  glm::dvec3 mid = pa + pb;  // sum first
  mid = glm::normalize(mid); // normalize immediately

  int idx = static_cast<int>(positions_.size());
  positions_.push_back(mid);
  (*cache)[key] = idx;
  return idx;
}
