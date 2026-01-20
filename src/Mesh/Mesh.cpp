#include "Mesh/Mesh.hpp"
#include "Polyhedron/Polyhedron.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <tetgen.h>
#include <utility>

void Mesh::generateFromWavefrontFile(const std::string &filename, double l0) {
  auto p = Polyhedron::fromWavefront(filename);
  generateFromPolyhedron(p, l0);
}

void Mesh::generateFromMshFile(const std::string &filename) {
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    throw std::runtime_error("[Mesh] Could not open msh file: " + filename);
  }

  vertices_.clear();
  tets_.clear();
  std::map<int, int> nodeIdToIndex; // Maps MSH Node ID -> std::vector index

  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty())
      continue;

    if (line == "$Nodes") {
      // Parse MSH 4.1 Nodes
      // Header: numEntityBlocks numNodes minNodeTag maxNodeTag
      if (!std::getline(ifs, line))
        break;
      std::istringstream iss(line);
      int numEntityBlocks, numNodes, minTag, maxTag;
      iss >> numEntityBlocks >> numNodes >> minTag >> maxTag;

      // Pre-allocate to avoid reallocations
      vertices_.reserve(numNodes);

      for (int b = 0; b < numEntityBlocks; ++b) {
        if (!std::getline(ifs, line))
          break;
        std::istringstream biss(line);
        int entityDim, entityTag, parametric, numNodesInBlock;
        biss >> entityDim >> entityTag >> parametric >> numNodesInBlock;

        // In MSH 4.1, a block lists ALL tags first, then ALL coordinates
        std::vector<int> blockNodeTags(numNodesInBlock);

        // Read Node Tags
        for (int i = 0; i < numNodesInBlock; ++i) {
          if (!std::getline(ifs, line))
            throw std::runtime_error("Unexpected EOF in Node Tags");
          blockNodeTags[i] = std::stoi(line);
        }

        // Read Node Coordinates
        for (int i = 0; i < numNodesInBlock; ++i) {
          if (!std::getline(ifs, line))
            throw std::runtime_error("Unexpected EOF in Node Coords");
          std::istringstream ciss(line);
          double x, y, z;
          ciss >> x >> y >> z;

          // Store vertex and map ID
          int idx = static_cast<int>(vertices_.size());
          nodeIdToIndex[blockNodeTags[i]] = idx;
          vertices_.emplace_back(x, y, z);
        }
      }
    } else if (line == "$Elements") {
      // Parse MSH 4.1 Elements
      // Header: numEntityBlocks numElements minElTag maxElTag
      if (!std::getline(ifs, line))
        break;
      std::istringstream iss(line);
      int numEntityBlocks, numElements, minElTag, maxElTag;
      iss >> numEntityBlocks >> numElements >> minElTag >> maxElTag;

      for (int b = 0; b < numEntityBlocks; ++b) {
        if (!std::getline(ifs, line))
          break;
        std::istringstream biss(line);
        int entityDim, entityTag, elementType, numElementsInBlock;
        biss >> entityDim >> entityTag >> elementType >> numElementsInBlock;

        // Gmsh Element Type 4 is a 4-node Tetrahedron
        if (elementType == 4) {
          for (int i = 0; i < numElementsInBlock; ++i) {
            if (!std::getline(ifs, line))
              break;
            std::istringstream elss(line);
            int tag, n1, n2, n3, n4;
            elss >> tag >> n1 >> n2 >> n3 >> n4;

            Tet tet;
            try {
              tet.vertids[0] = nodeIdToIndex.at(n1);
              tet.vertids[1] = nodeIdToIndex.at(n2);
              tet.vertids[2] = nodeIdToIndex.at(n3);
              tet.vertids[3] = nodeIdToIndex.at(n4);
              tets_.push_back(tet);
            } catch (const std::out_of_range &) {
              throw std::runtime_error(
                  "[Mesh] Element references unknown node id: " +
                  std::to_string(n1));
            }
          }
        } else {
          // Skip lines for non-tet elements
          for (int i = 0; i < numElementsInBlock; ++i) {
            std::getline(ifs, line);
          }
        }
      }
    }
  }

  if (vertices_.empty() || tets_.empty()) {
    throw std::runtime_error(
        "[Mesh] Failed to read vertices or tets from msh.");
  }

  std::cout << "[Mesh] (" << vertices_.size() << " vertices, " << tets_.size()
            << " tets) loaded from " << filename << std::endl;

  buildConnectivity();
  computeGeometry();
  ensureConsistentFaceNormals();
  computeShapeFunctionGradients();
  buildP2EdgeNodes();
  computeMinEdgeLength();
  computeMinEdgeLength();

  solid_vert_bc_types_.resize(vertices_.size(), SolidBCType::Undefined);
  p1_fluid_vert_bc_types_.resize(vertices_.size(), FluidBCType::Undefined);

  if (hasDegenerateTet()) {
    throw std::runtime_error("[Mesh] degenerate tetrahedral elements found.\n");
  }
}

void Mesh::generateStructuredRectangularPrism(double length, double width,
                                              double height, int nx, int ny,
                                              int nz) {
  if (nx < 1 || ny < 1 || nz < 1) {
    throw std::invalid_argument("nx, ny, nz must be >= 1");
  }

  const int nxp = nx + 1;
  const int nyp = ny + 1;
  const int nzp = nz + 1;

  const double dx = length / static_cast<double>(nx);
  const double dy = width / static_cast<double>(ny);
  const double dz = height / static_cast<double>(nz);

  auto idx = [nxp, nyp](int i, int j, int k) {
    return i + nxp * (j + nyp * k);
  };

  // start from empty mesh
  vertices_.clear();
  tets_.clear();

  vertices_.reserve(static_cast<size_t>(nxp) * nyp * nzp);
  for (int k = 0; k < nzp; ++k) {
    for (int j = 0; j < nyp; ++j) {
      for (int i = 0; i < nxp; ++i) {
        double x = static_cast<double>(i) * dx;
        double y = static_cast<double>(j) * dy;
        double z = static_cast<double>(k) * dz;
        vertices_.emplace_back(x, y, z);
      }
    }
  }

  // Reserve estimated number of tets: 6 per cell
  tets_.reserve(static_cast<size_t>(nx) * ny * nz * 6);

  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        // local cube corners
        int v0 = idx(i, j, k);             // a: 0,0,0
        int v1 = idx(i + 1, j, k);         // b: 1,0,0
        int v2 = idx(i, j + 1, k);         // c: 0,1,0
        int v3 = idx(i + 1, j + 1, k);     // d:1,1,0
        int v4 = idx(i, j, k + 1);         // e:0,0,1
        int v5 = idx(i + 1, j, k + 1);     // f:1,0,1
        int v6 = idx(i, j + 1, k + 1);     // g:0,1,1
        int v7 = idx(i + 1, j + 1, k + 1); // h:1,1,1

        // Use a consistent body-diagonal (v0 - v7) for all cells so
        // faces between adjacent cells are triangulated identically.
        Tet t;
        t.vertids = {v0, v1, v3, v7};
        tets_.push_back(t);
        t.vertids = {v0, v3, v2, v7};
        tets_.push_back(t);
        t.vertids = {v0, v2, v6, v7};
        tets_.push_back(t);
        t.vertids = {v0, v6, v4, v7};
        tets_.push_back(t);
        t.vertids = {v0, v4, v5, v7};
        tets_.push_back(t);
        t.vertids = {v0, v5, v1, v7};
        tets_.push_back(t);
      }
    }
  }

  buildConnectivity();
  computeGeometry();
  ensureConsistentFaceNormals();
  computeShapeFunctionGradients();
  buildP2EdgeNodes();
  computeMinEdgeLength();
  solid_vert_bc_types_.resize(vertices_.size(), SolidBCType::Undefined);
  p1_fluid_vert_bc_types_.resize(vertices_.size(), FluidBCType::Undefined);
  assert(!hasDegenerateTet());
}

void Mesh::generateFromPolyhedron(const Polyhedron &poly, double l0) {
  tetgenio in, out;
  in.initialize();
  out.initialize();

  in.mesh_dim = 3;
  in.numberofpoints = static_cast<int>(poly.nVerts());
  in.pointlist = new REAL[in.numberofpoints * 3];

  for (int i = 0; i < in.numberofpoints; i++) {
    const Eigen::Vector3d p = poly.getPosition(i);
    in.pointlist[3 * i + 0] = static_cast<REAL>(p.x());
    in.pointlist[3 * i + 1] = static_cast<REAL>(p.y());
    in.pointlist[3 * i + 2] = static_cast<REAL>(p.z());
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
  computeGeometry();
  ensureConsistentFaceNormals();
  computeShapeFunctionGradients();
  buildP2EdgeNodes();
  computeMinEdgeLength();
  solid_vert_bc_types_.resize(vertices_.size(), SolidBCType::Undefined);
  p1_fluid_vert_bc_types_.resize(vertices_.size(), FluidBCType::Undefined);
  if (hasDegenerateTet()) {
    throw std::runtime_error("[Mesh] degenerate tetrahedral elements found.\n");
  }
}

void Mesh::setupBoundaryConditions(
    const Eigen::Vector3d &inlet_to_outlet_direction, double percent_coverage,
    Mesh &mesh) {
  if (mesh.nFaces() == 0 || mesh.nVertices() < 3) {
    return;
  }

  std::cout << "[Mesh][BC] Applying boundary conditions to mesh... "
            << std::flush;

  int n_wall = 0;
  int n_inlet = 0;
  int n_outlet = 0;

  Eigen::Vector3d min_bounds = mesh.getVertexPositon(0);
  Eigen::Vector3d max_bounds = mesh.getVertexPositon(0);

  for (size_t vi = 0; vi < mesh.nVertices(); vi++) {
    min_bounds = min_bounds.cwiseMin(mesh.getVertexPositon(vi));
    max_bounds = max_bounds.cwiseMax(mesh.getVertexPositon(vi));
  }

  const Eigen::Vector3d mesh_size = max_bounds - min_bounds;
  int primary_axis = 0;
  if (mesh_size.y() > mesh_size.x() && mesh_size.y() > mesh_size.z())
    primary_axis = 1;
  else if (mesh_size.z() > mesh_size.x() && mesh_size.z() > mesh_size.y())
    primary_axis = 2;

  // Boundary assignment params
  const double tolerance = mesh_size[primary_axis] * 0.01 * percent_coverage;
  ;
  const double inlet_pos = inlet_to_outlet_direction[primary_axis] > 0
                               ? min_bounds[primary_axis]
                               : max_bounds[primary_axis];
  const double outlet_pos = inlet_to_outlet_direction[primary_axis] > 0
                                ? max_bounds[primary_axis]
                                : min_bounds[primary_axis];

  for (size_t vi = 0; vi < mesh.nVertices(); vi++) {
    // set all boundaries to internal.
    mesh.setFluidVertexBC(vi, FluidBCType::Internal);
    mesh.setSolidVertexBC(vi, SolidBCType::Free);
  }

  const std::vector<Face> &faces = mesh.getFaces();
  for (size_t fi = 0; fi < mesh.nFaces(); fi++) {
    if (mesh.isFaceInternal(fi)) {
      mesh.faces_[fi].is_ecm = false;
      continue; // skip internal faces
    }
    mesh.faces_[fi].is_ecm = true;
    // if wall, set to wall
    for (const int &vi : faces[fi].vertids) {
      mesh.setFluidVertexBC(vi, FluidBCType::Wall);
      n_wall++;
    }

    // assign inlet and outlet based on position and face normal with respect to
    // flow direction
    double dist_to_inlet = std::abs(faces[fi].center[primary_axis] - inlet_pos);
    double dist_to_outlet =
        std::abs(faces[fi].center[primary_axis] - outlet_pos);

    if (dist_to_inlet < tolerance) {
      mesh.faces_[fi].is_ecm = false;
      for (const auto &vi : faces[fi].vertids) {
        mesh.setFluidVertexBC(vi, FluidBCType::Inlet);
        mesh.setSolidVertexBC(vi, SolidBCType::Fixed);
        n_inlet++;
        n_wall--;
      }
    } else if (dist_to_outlet < tolerance) {
      mesh.faces_[fi].is_ecm = false;
      for (const int &vi : faces[fi].vertids) {
        mesh.setFluidVertexBC(vi, FluidBCType::Outlet);
        mesh.setSolidVertexBC(vi, SolidBCType::Fixed);
        n_outlet++;
        n_wall--;
      }
    }
  }
  // set p2 data
  mesh.setP2BoundariesFromP1Boundaries();
  std::cout << "Competed.\n";
  std::cout << "[Mesh][BC] NavierStokes boundaries:\n";
  std::cout << "\tInlet BC:" << n_inlet << " Outlet BC:" << n_outlet
            << " Wall BC:" << n_wall << "\n";
  std::cout << "[Mesh][BC] Solid mechanic boundaries:\n";
  std::cout << "\tFixed BC:" << n_outlet + n_inlet
            << " Free BC:" << mesh.nVertices() - n_outlet - n_inlet
            << std::endl;
}

void Mesh::computeMinEdgeLength() {
  double min_len_sq = std::numeric_limits<double>::max();
  for (const auto &tet : tets_) {
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {

        int idx_a = tet.vertids[i];
        int idx_b = tet.vertids[j];

        const Eigen::Vector3d &p_a = vertices_[idx_a];
        const Eigen::Vector3d &p_b = vertices_[idx_b];

        Eigen::Vector3d diff = p_a - p_b;
        double len_sq = diff.dot(diff);

        if (len_sq < min_len_sq) {
          min_len_sq = len_sq;
        }
      }
    }
  }

  if (min_len_sq == std::numeric_limits<double>::max()) {
    min_edge_length_ = 0.0;
  } else {
    min_edge_length_ = std::sqrt(min_len_sq);
  }
}

void Mesh::computeGeometry() {
  for (auto &tet : tets_) {
    const Eigen::Vector3d &a = vertices_[tet.vertids[0]];
    const Eigen::Vector3d &b = vertices_[tet.vertids[1]];
    const Eigen::Vector3d &c = vertices_[tet.vertids[2]];
    const Eigen::Vector3d &d = vertices_[tet.vertids[3]];
    tet.centroid = (a + b + c + d) / 4.0;
    tet.volume = std::fabs((b - a).cross(c - a).dot(d - a) / 6.0);
    if (tet.volume <= 0.0) {
      std::cerr << "[Mesh][Warning] Degenerate Tet found." << std::endl;
    }
  }
  for (auto &face : faces_) {
    const Eigen::Vector3d &a = vertices_[face.vertids[0]];
    const Eigen::Vector3d &b = vertices_[face.vertids[1]];
    const Eigen::Vector3d &c = vertices_[face.vertids[2]];
    const Eigen::Vector3d ab = b - a;
    const Eigen::Vector3d ac = c - a;
    face.normal = ac.cross(ab).normalized();
    face.area = 0.5 * ab.cross(ac).norm();
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
  buildVertexNeighbors();
}

void Mesh::computeShapeFunctionGradients() {
  tet_gradients_.resize(tets_.size());
  for (size_t ti = 0; ti < tets_.size(); ti++) {
    const auto &tet = tets_[ti];
    const Eigen::Vector3d &p0 = vertices_[tet.vertids[0]];
    const Eigen::Vector3d &p1 = vertices_[tet.vertids[1]];
    const Eigen::Vector3d &p2 = vertices_[tet.vertids[2]];
    const Eigen::Vector3d &p3 = vertices_[tet.vertids[3]];

    Eigen::Vector3d e1 = p1 - p0;
    Eigen::Vector3d e2 = p2 - p0;
    Eigen::Vector3d e3 = p3 - p0;

    double sixv = e1.dot(e2.cross(e3));
    double invsixv = 1.0 / sixv;

    // gradients for linear tetrahedral elements
    std::array<Eigen::Vector3d, 4> grad_n;
    grad_n[0] = invsixv * (p3 - p1).cross(p2 - p1);
    grad_n[1] = invsixv * (p2 - p0).cross(p3 - p0);
    grad_n[2] = invsixv * (p3 - p0).cross(p1 - p0);
    grad_n[3] = invsixv * (p1 - p0).cross(p2 - p0);
    tet_gradients_[ti] = grad_n;
  }
}
void Mesh::ensureConsistentFaceNormals() {
  const double eps = 1e-12;
  for (auto &face : faces_) {
    if (face.tet_a >= 0 && face.tet_b >= 0) {
      // Internal face: normal should point from owner (tet_a) to neighbour
      // (tet_b)
      const Eigen::Vector3d &centroid_a = tets_[face.tet_a].centroid;
      const Eigen::Vector3d &centroid_b = tets_[face.tet_b].centroid;
      Eigen::Vector3d desired_direction =
          (centroid_b - centroid_a).normalized();
      if (face.normal.dot(desired_direction) <= eps) {
        face.normal = -face.normal;
        std::swap(face.vertids[1], face.vertids[2]);
      } else if (face.tet_a >= 0) {
        const Eigen::Vector3d &centroid_a = tets_[face.tet_a].centroid;
        const Eigen::Vector3d outward_dir =
            (face.center - centroid_a).normalized();
        if (face.normal.dot(outward_dir) <= eps) {
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
        Eigen::Vector3d midpoint = 0.5 * (vertices_[v1] + vertices_[v2]);
        edge_nodes_.push_back(midpoint);
        edge_to_node_id_[edge_key] = edge_node_id;
        tet_edge_nodes_[tet_idx][e] = edge_node_id;
        edge_node_id++;
      } else {
        tet_edge_nodes_[tet_idx][e] = edge_to_node_id_[edge_key];
      }
    }
  }
  p2_fluid_vert_bc_types_.resize(edge_nodes_.size(), FluidBCType::Undefined);
  std::cout << "[Mesh] P2 elements: Created " << edge_nodes_.size()
            << " edge nodes for P2-P1 Taylor-Hood dissscretization"
            << std::endl;
}
bool Mesh::hasDegenerateTet() {
  for (const auto &tet : tets_) {
    if (tet.volume <= 0.0) {
      return true;
    }
  }
  return false;
}
