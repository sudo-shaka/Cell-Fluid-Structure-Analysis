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
  double volume = 0.0;
  std::array<int, 4> vertids;
  std::array<int, 4> faceids = {-1, -1, -1, -1};
  std::array<int, 4> neighbors = {-1, -1, -1, -1};
  glm::dvec3 centroid = glm::dvec3(0);
};

struct Face {
  bool is_ecm = false; // if true deforable particles can adhere to this face.
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
  std::vector<FluidBCType> p1_fluid_vert_bc_types_;
  std::vector<FluidBCType> p2_fluid_vert_bc_types_;
  std::vector<SolidBCType> solid_vert_bc_types_;
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
  const std::vector<Tet> &getTets() const { return tets_; }
  const Tet &tetAt(const size_t tetidx) const {
    assert(tetidx < tets_.size());
    return tets_[tetidx];
  }
  Tet &mutTetAt(const size_t tetidx) {
    assert(tetidx < tets_.size());
    return tets_[tetidx];
  }
  const std::vector<Face> &getFaces() const { return faces_; }
  const Face &faceAt(const size_t faceidx) const {
    assert(faceidx < faces_.size());
    return faces_[faceidx];
  }
  Face &mutFaceAt(const size_t faceidx) {
    assert(faceidx < faces_.size());
    return faces_[faceidx];
  }
  const std::vector<glm::dvec3> &getVertPositions() const { return vertices_; }
  const glm::dvec3 &getVertexPositon(const size_t vertid) const {
    assert(vertid < vertices_.size());
    return vertices_[vertid];
  }
  glm::dvec3 &getMutVertexPositon(const size_t vertid) {
    assert(vertid < vertices_.size());
    return vertices_[vertid];
  }
  FluidBCType getFluidVertexBC(const size_t vertex_id) const {
    assert(vertex_id < p1_vertex_bc_types_.size());
    return p1_fluid_vert_bc_types_[vertex_id];
  }
  SolidBCType getSolidVertexBC(const size_t vid) const {
    assert(vid < solid_vert_bc_types_[vid]);
    return solid_vert_bc_types_[vid];
  }
  const std::vector<int> &getVertexNeighborInds(size_t vi) const {
    assert(vi < vertices_.size());
    return vertex_neighbors_[vi];
  }
  const glm::dvec3 &getP2NodesAtEdge(size_t vi, size_t vj) const {
    auto key = std::pair(std::min(vi, vj), std::max(vi, vj));
    auto it = edge_to_node_id_.find(key);
    assert(it != edge_to_node_id_.end());
    return edge_nodes_[it->second];
  }

  // setters
  void setFluidVertexBC(const size_t vertex_id, const FluidBCType bc_type) {
    assert(vertex_id < p1_vertex_bc_types_.size());
    p1_fluid_vert_bc_types_[vertex_id] = bc_type;
  }
  void setFluidP2vertexBC(const size_t p2_id, const FluidBCType bc_type) {
    assert(p2_id < p2_vertex_bc_types_.size());
    p2_fluid_vert_bc_types_[p2_id] = bc_type;
  }
  void setSolidP2vertexBC(const size_t vid, const SolidBCType bc_type) {
    assert(vid < SolidBCType);
    solid_vert_bc_types_[vid] = bc_type;
  }
  bool isFaceInternal(const size_t fi) const {
    assert(fi < faces_.size());
    return faces_[fi].tet_b != -1;
  }
  bool isFaceWall(const size_t fi) const {
    assert(fi < faces_.size());
    if (isFaceInternal(fi)) {
      return false;
    }
    const auto &f = faces_[fi].vertids;
    if (p1_fluid_vert_bc_types_[f[0]] == FluidBCType::Wall &&
        p1_fluid_vert_bc_types_[f[1]] == FluidBCType::Wall &&
        p1_fluid_vert_bc_types_[f[2]] == FluidBCType::Wall) {
      return true;
    }
    return false;
  }
};
