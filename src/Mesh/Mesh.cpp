#include "Mesh/Mesh.hpp"
#include "Polyhedron/Polyhedron.hpp"
#include <algorithm>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <iostream>
#include <tetgen.h>
#include <utility>

void Mesh::generateFromWavefrontFile(const std::string &filename, double l0) {
  auto p = Polyhedron::fromWavefront(filename);
  generateFromPolyhedron(p, l0);
}

void Mesh::generateFromMshFile(const std::string &filename) { /*TODO*/ }

void Mesh::generateFromPolyhedron(const Polyhedron &poly, double l0) {
  tetgenio in, out;
  in.initialize();
  out.initialize();

  in.mesh_dim = 3;
  in.numberofpoints = static_cast<int>(poly.nVerts());
  in.pointlist = new REAL[in.numberofpoints * 3];

  for (int i = 0; i < in.numberofpoints; i++) {
    const glm::dvec3 p = poly.getPosition(i);
    in.pointlist[3 * i + 0] = static_cast<REAL>(p.x);
    in.pointlist[3 * i + 1] = static_cast<REAL>(p.y);
    in.pointlist[3 * i + 2] = static_cast<REAL>(p.z);
  }
  in.numberoffacets = static_cast<int>(poly.nFaces());
  // Value-initialize facets/polygons to avoid uninitialized fields
  in.facetlist = new tetgenio::facet[in.numberoffacets]();
  for (int f = 0; f < in.numberoffacets; f++) {
    tetgenio::facet *facet = &in.facetlist[f];
    facet->numberofpolygons = 1;
    facet->polygonlist = new tetgenio::polygon[1]();
    facet->numberofholes = 0;
    facet->holelist = nullptr;

    tetgenio::polygon *p = &facet->polygonlist[0];
    p->numberofvertices = 3;
    p->vertexlist = new int[3];

    // assuming Matrix.faces[f] is glm::ivec3 or similar
    const auto &matrix_faces = poly.getFaceIndices(f);
    p->vertexlist[0] = matrix_faces[0];
    p->vertexlist[1] = matrix_faces[1];
    p->vertexlist[2] = matrix_faces[2];
  }

  tetgenbehavior b;

  // Diagnostic: print input point count and bounding box to help debug
  if (in.numberofpoints > 0) {
    double xmin = in.pointlist[0];
    double xmax = in.pointlist[0];
    double ymin = in.pointlist[1];
    double ymax = in.pointlist[1];
    double zmin = in.pointlist[2];
    double zmax = in.pointlist[2];
    for (int i = 0; i < in.numberofpoints; ++i) {
      double px = static_cast<double>(in.pointlist[3 * i + 0]);
      double py = static_cast<double>(in.pointlist[3 * i + 1]);
      double pz = static_cast<double>(in.pointlist[3 * i + 2]);
      xmin = std::min(xmin, px);
      xmax = std::max(xmax, px);
      ymin = std::min(ymin, py);
      ymax = std::max(ymax, py);
      zmin = std::min(zmin, pz);
      zmax = std::max(zmax, pz);
    }
  }

  // p = use surface PLC
  // q = quality constraint
  // a0.0 = no max volume (can set e.g. a0.01 for finer mesh)
  // z = zero-based indexing
  // Q = quiet

  char temp[32];
  snprintf(temp, sizeof(temp), "%.16g", l0);
  std::string arg_string = std::string("pq2.2a") + temp + "zQ";
  const char *args = arg_string.c_str();
  std::cout << "[Mesh] TetGen args: " << args << std::endl;
  b.parse_commandline(const_cast<char *>(args));
  b.zeroindex = 1;

  tetrahedralize(&b, &in, &out);

  if (out.numberofpoints <= 0 || out.numberoftetrahedra <= 0) {
    throw std::runtime_error("[Mesh] TetGen produced empty volume mesh.");
  }

  vertices_.clear();
  vertices_.reserve(out.numberofpoints);
  for (int i = 0; i < out.numberofpoints; ++i) {
    REAL *p = &out.pointlist[3 * i];
    vertices_.emplace_back(static_cast<double>(p[0]), static_cast<double>(p[1]),
                           static_cast<double>(p[2]));
  }

  tets_.clear();
  tets_.reserve(out.numberoftetrahedra);
  const int n_corners = (out.numberofcorners > 0) ? out.numberofcorners : 4;
  for (int t = 0; t < out.numberoftetrahedra; ++t) {
    Tet tet;
    const int base = t * n_corners;
    tet.vertids[0] = out.tetrahedronlist[base + 0];
    tet.vertids[1] = out.tetrahedronlist[base + 1];
    tet.vertids[2] = out.tetrahedronlist[base + 2];
    tet.vertids[3] = out.tetrahedronlist[base + 3];
    tets_.push_back(tet);
  }

  std::cout << "[Mesh] (" << out.numberofpoints << " vertices, "
            << out.numberoftetrahedra << " tets)" << std::endl;

  delete[] in.pointlist;
  in.pointlist = nullptr;

  buildConnectivity();
  ensureConsistentFaceNormals();
  computeGeometry();
  computeShapeFunctionGradients();
  buildP2EdgeNodes();
  if (hasDegenerateTet()) {
    throw std::runtime_error("[Mesh] degenerate tetrahedral elements found.\n");
  }
}

void Mesh::computeGeometry() {
  for (auto &tet : tets_) {
    const glm::dvec3 &a = vertices_[tet.vertids[0]];
    const glm::dvec3 &b = vertices_[tet.vertids[0]];
    const glm::dvec3 &c = vertices_[tet.vertids[0]];
    const glm::dvec3 &d = vertices_[tet.vertids[0]];
    tet.centroid = (a + b + c + d) / 4.0;
    tet.volume = std::fabs(glm::dot(glm::cross(b - a, c - a), d - a) / 6.0);
    if (tet.volume <= 0.0) {
      std::cerr << "[Mesh][Warning] Degenerate Tet found.\n" << std::endl;
    }
  }
  for (auto &face : faces_) {
    const glm::dvec3 &a = vertices_[face.vertids[0]];
    const glm::dvec3 &b = vertices_[face.vertids[1]];
    const glm::dvec3 &c = vertices_[face.vertids[2]];
    const glm::dvec3 ab = b - a;
    const glm::dvec3 ac = c - a;
    face.normal = glm::normalize(glm::cross(ac, ab));
    face.area = 0.5 * glm::length(glm::cross(ab, ac));
    face.center = (a + b + c) / 3.0;
  }
  computeShapeFunctionGradients();
}
void Mesh::buildVertexNeighbors() {
  size_t n_verts = vertices_.size();
  vertex_neighbors_.resize(n_verts);
  for (const auto &tet : tets_) {
    for (size_t i = 0; i < tet.vertids.size(); i++) {
      int vi = tet.vertids[i];
      for (size_t j = 0; j < tet.vertids.size(); j++) {
        if (i == j) {
          continue;
        }
        int vj = tet.vertids[j];

        // avoid duplicates
        if (std::find(vertex_neighbors_[vi].begin(),
                      vertex_neighbors_[vi].end(),
                      vj) == vertex_neighbors_[vi].end()) {
          vertex_neighbors_[vi].push_back(vj);
        }
      }
    }
  }
};
void Mesh::buildConnectivity() {
  faces_.clear();
  std::map<std::array<int, 3>, Face> face_map;

  const std::array<std::array<int, 3>, 4> local_face_indices = {{
      {{1, 2, 3}},
      {{0, 3, 2}},
      {{0, 1, 3}},
      {{0, 2, 1}},
  }};

  // First pass: create faces without orientation checking
  for (int tet_idx = 0; tet_idx < static_cast<int>(tets_.size()); ++tet_idx) {
    Tet &tet = tets_[tet_idx];
    for (int face_id = 0; face_id < 4; ++face_id) {
      const auto &lf = local_face_indices[face_id];
      const auto &vid = tet.vertids;
      std::array<int, 3> face_verts = {vid[lf[0]], vid[lf[1]], vid[lf[2]]};
      std::array<int, 3> sorted_verts = face_verts;
      std::sort(sorted_verts.begin(), sorted_verts.end());
      auto it = face_map.find(sorted_verts);
      if (it == face_map.end()) {
        Face face;
        face.vertids = face_verts;
        face.tet_a = tet_idx;
        face.face_id_a = face_id;
        face_map[sorted_verts] = face;
      } else {
        it->second.tet_b = tet_idx;
        it->second.face_id_b = face_id;
      }
    }
  }

  for (auto &[_, face] : face_map) {
    faces_.push_back(face);
  }

  // Build face ID mappings
  auto get_tet_face_verts = [](const Tet &tet,
                               int face_index) -> std::array<int, 3> {
    const std::array<std::array<int, 3>, 4> local_faces = {{
        {{1, 2, 3}}, // opposite to vertex 0
        {{0, 3, 2}}, // opposite to vertex 1
        {{0, 1, 3}}, // opposite to vertex 2
        {{0, 2, 1}}, // opposite to vertex 3
    }};

    std::array<int, 3> face_verts;
    for (int i = 0; i < 3; ++i) {
      face_verts[i] = tet.vertids[local_faces[face_index][i]];
    }
    return face_verts;
  };

  // Build a map from face vertex signature -> face index in mesh.faces
  std::map<std::array<int, 3>, int> faceVertToIndex;
  for (size_t f = 0; f < faces_.size(); ++f) {
    std::array<int, 3> sortedVerts = faces_[f].vertids;
    std::sort(sortedVerts.begin(), sortedVerts.end());
    faceVertToIndex[sortedVerts] = static_cast<int>(f);
  }

  // Assign face_ids to each tet
  for (size_t t = 0; t < tets_.size(); ++t) {
    Tet &tet = tets_[t];
    for (int f = 0; f < 4; ++f) {
      std::array<int, 3> faceVerts = get_tet_face_verts(tet, f);
      std::array<int, 3> sortedVerts = faceVerts;
      std::sort(sortedVerts.begin(), sortedVerts.end());

      auto it = faceVertToIndex.find(sortedVerts);
      if (it != faceVertToIndex.end()) {
        tet.faceids[f] = it->second;
      }
    }
  }

  // Build neighbor connectivity with validation
  for (const auto &face : faces_) {
    if (face.tet_a >= 0 && face.tet_b >= 0) {
      Tet &tet_a = tets_[face.tet_a];
      Tet &tet_b = tets_[face.tet_b];

      int fA = face.face_id_a;
      int fB = face.face_id_b;

      if (fA >= 0 && fA < 4 && fB >= 0 && fB < 4) {
        tet_a.neighbors[fA] = face.tet_b;
        tet_b.neighbors[fB] = face.tet_a;
      }
    }
  }
  buildIncidentTets();
  buildVertexNeighbors();
}
void Mesh::buildIncidentTets() {
  vertex_incident_tets_.resize(vertices_.size());
  for (size_t ti = 0; ti < tets_.size(); ti++) {
    const auto &tet = tets_[ti];
    for (const int &vid : tet.vertids) {
      if (vid >= 0 && vid < static_cast<int>(vertices_.size())) {
        vertex_incident_tets_[vid].push_back(static_cast<int>(ti));
      }
    }
  }
}
void Mesh::computeShapeFunctionGradients() {
  tet_gradients_.resize(tets_.size());
  for (size_t ti = 0; ti < tets_.size(); ti++) {
    const auto &tet = tets_[ti];
    const glm::dvec3 &p0 = vertices_[tet.vertids[0]];
    const glm::dvec3 &p1 = vertices_[tet.vertids[1]];
    const glm::dvec3 &p2 = vertices_[tet.vertids[2]];
    const glm::dvec3 &p3 = vertices_[tet.vertids[3]];

    glm::dvec3 e1 = p1 - p0;
    glm::dvec3 e2 = p2 - p0;
    glm::dvec3 e3 = p3 - p0;

    double sixv = glm::dot(e1, glm::cross(e2, e3));
    double invsixv = 1.0 / sixv;

    // gradients for linear tetrahedral elements
    std::array<glm::dvec3, 4> grad_n;
    grad_n[0] = invsixv * glm::cross(p2 - p1, p3 - p1);
    grad_n[1] = invsixv * glm::cross(p3 - p0, p2 - p0);
    grad_n[2] = invsixv * glm::cross(p1 - p0, p3 - p0);
    grad_n[3] = invsixv * glm::cross(p2 - p0, p1 - p0);
    tet_gradients_[ti] = grad_n;
  }
}
void Mesh::ensureConsistentFaceNormals() {
  const double eps = 1e-12;
  for (auto &face : faces_) {
    if (face.tet_a >= 0 && face.tet_b >= 0) {
      // Internal face: normal should point from owner (tet_a) to neighbour
      // (tet_b)
      const glm::dvec3 &centroid_a = tets_[face.tet_a].centroid;
      const glm::dvec3 &centroid_b = tets_[face.tet_b].centroid;
      glm::dvec3 desired_direction = glm::normalize(centroid_b - centroid_a);
      if (glm::dot(face.normal, desired_direction) <= eps) {
        face.normal = -face.normal;
        std::swap(face.vertids[1], face.vertids[2]);
      } else if (face.tet_a >= 0) {
        const glm::dvec3 &centroid_a = tets_[face.tet_a].centroid;
        const glm::dvec3 outward_dir = glm::normalize(face.center - centroid_a);
        if (glm::dot(face.normal, outward_dir) <= eps) {
          face.normal = -face.normal;
          std::swap(face.vertids[1], face.vertids[2]);
        }
      }
    }
  }
}

void Mesh::buildP2EdgeNodes() {
  edge_nodes_.clear();
  edge_to_node_id_.clear();
  tet_edge_nodes_.clear();
  tet_edge_nodes_.resize(tets_.size());
  int edge_node_id = 0;

  for (size_t tet_idx = 0; tet_idx < tets_.size(); tet_idx++) {
    const auto &tet = tets_[tet_idx];
    const auto &verts = tet.vertids;
    // 6 edges of a tetrahedron in standard order: (0-1, 0-2, 0-3, 1-2, 1-3,
    // 2-3)
    int edge_pairs[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    for (int e = 0; e < 6; e++) {
      int v1 = verts[edge_pairs[e][0]];
      int v2 = verts[edge_pairs[e][1]];
      if (v1 > v2) {
        std::swap(v1, v2);
      }
      auto edge_key = std::make_pair(v1, v2);
      if (edge_to_node_id_.find(edge_key) == edge_to_node_id_.end()) {
        glm::dvec3 midpoint = 0.5 * (vertices_[v1] + vertices_[v2]);
        edge_nodes_.push_back(midpoint);
        edge_to_node_id_[edge_key] = edge_node_id;
        tet_edge_nodes_[tet_idx][e] = edge_node_id;
        edge_node_id++;
      } else {
        tet_edge_nodes_[tet_idx][e] = edge_to_node_id_[edge_key];
      }
    }
  }
  std::cout << "[Mesh] P2 elements: Created " << edge_nodes_.size()
            << " edge nodes for P2-P1 Taylor-Hood dissscretization"
            << std::endl;
}
bool Mesh::hasDegenerateTet() {
  for (const auto &tet : tets_) {
    if (tet.volume <= 0.0) {
      return false;
    }
  }
  return true;
}
