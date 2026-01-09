#pragma once

#include "BC/BC.hpp"
#include "Polyhedron/Polyhedron.hpp"
#include <array>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <map>
#include <string>
#include <vector>

class DeformableParticle;

struct Tet {
  BoundaryType bc = BoundaryType::Undefined;
  double volume = 0.0;
  std::array<int, 4> vertids;
  std::array<int, 4> faceids = {-1, -1, -1, -1};
  std::array<int, 4> neighbors = {-1, -1, -1, -1};
  glm::dvec3 centroid = glm::dvec3(0);
};

struct Face {
  BoundaryType bc = BoundaryType::Undefined;
  int tet_a = -1;
  int tet_b = -1;
  int face_id_a = -1;
  int face_id_b = -1;
  double area = 0.0;
  std::array<int, 3> vertids;
  glm::dvec3 normal = glm::dvec3(0);
  glm::dvec3 center;
};

class Mesh {
  std::vector<glm::dvec3> vertices_;   // P1 Elements
  std::vector<glm::dvec3> edge_nodes_; // P2 Elents (Taylor Hood)
  std::vector<Face> faces_;
  std::vector<Tet> tets_;
  std::vector<std::vector<int>> vertex_neighbors_;
  std::vector<std::vector<int>> vertex_incident_tets_;
  std::vector<BoundaryType> p1_vertex_bc_types_;
  std::vector<BoundaryType> p2_vert_bc_types_;
  std::vector<std::array<glm::dvec3, 4>> tet_gradients_;
  // Map from edge (vi, vj) to edge node index. Key: (min(vi,vj), max(vi,vj))
  std::map<std::pair<int, int>, int> edge_to_node_id_;
  std::vector<std::array<int, 6>> tet_edge_nodes_;

  Mesh() = default;
  void generateFromWavefrontFile(const std::string &filename,
                                 double max_edge_length);
  void generateFromPolyhedron(const Polyhedron &polyhedron,
                              double max_edge_length);
  void generateFromMshFile(const std::string &filename); // TODO

  void computeGeometry();
  void buildVertexNeighbors();
  void buildConnectivity();
  void buildIncidentTets();
  void computeShapeFunctionGradients();
  void ensureConsistentFaceNormals();
  void buildP2EdgeNodes();
  bool hasDegenerateTet();

public:
  static Mesh fromObjFile(const std::string &filename,
                          double max_edge_length = 0.1) {
    Mesh m;
    m.generateFromWavefrontFile(filename, max_edge_length);
    return m;
  }
  static Mesh fromPolyhedron(const Polyhedron &poly,
                             double max_edge_length = 0.1) {
    Mesh m;
    m.generateFromPolyhedron(poly, max_edge_length);
    return m;
  }
  static Mesh fromMshFile(const std::string &filename) {
    Mesh m;
    m.generateFromMshFile(filename);
    return m;
  }
  bool isInitialized() { return !vertices_.empty() && !tets_.empty(); }

  // sizes
  size_t nVertices() const { return vertices_.size(); }
  size_t nTets() const { return tets_.size(); }
  size_t nFaces() const { return faces_.size(); }

  // getters
  const Tet &tetAt(const size_t tetidx) const {
    assert(tetidx < tets_.size());
    return tets_[tetidx];
  }
  Tet &mutTetAt(const size_t tetidx) {
    assert(tetidx < tets_.size());
    return tets_[tetidx];
  }
  const Face &faceAt(const size_t faceidx) const {
    assert(faceidx < faces_.size());
    return faces_[faceidx];
  }
  Face &mutFaceAt(const size_t faceidx) {
    assert(faceidx < faces_.size());
    return faces_[faceidx];
  }
  // setters
};
