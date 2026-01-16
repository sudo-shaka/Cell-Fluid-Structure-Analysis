#pragma once

#include "LinearAlgebra/LinearSolvers.hpp"
#include "LinearAlgebra/SparseMatrix.hpp"
#include <array>
#include <functional>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <memory>
#include <vector>

// Forward-declare Mesh to avoid circular include
class Mesh;

/// Boundary condition type for solid mechanics
enum class SolidBCType {
  /// Free (natural BC - zero traction)
  Free,
  /// Fixed displacement (Dirichlet)
  Fixed,
  /// Prescribed displacement
  Displacement,
  Undefined,
};

/// Material model type
enum class MaterialModel {
  /// Linear elastic (Hooke's law)
  LinearElastic,
  /// Neo-Hookean hyperelastic (large deformations)
  NeoHookean,
  /// Mooney-Rivlin hyperelastic
  MooneyRivlin,
  /// Saint Venant-Kirchhoff (geometric nonlinearity only)
  StVenantKirchhoff
};

/// Material properties for solid mechanics
struct Material {
  MaterialModel model = MaterialModel::LinearElastic;

  /// Young's modulus [Pa]
  double youngs_modulus = 1e6;

  /// Poisson's ratio [-]
  double poissons_ratio = 0.3;

  /// Density [kg/m³]
  double density = 1000.0;

  /// Shear modulus (computed from E, nu)
  double shear_modulus = 0.0;

  /// Bulk modulus (computed from E, nu)
  double bulk_modulus = 0.0;

  /// First Lamé parameter
  double lambda = 0.0;

  /// Second Lamé parameter (= shear modulus)
  double mu = 0.0;

  /// Mooney-Rivlin C10 coefficient (for MooneyRivlin model)
  double c10 = 0.0;

  /// Mooney-Rivlin C01 coefficient (for MooneyRivlin model)
  double c01 = 0.0;

  /// Damping coefficient (Rayleigh mass damping)
  double alpha_m = 0.0;

  /// Damping coefficient (Rayleigh stiffness damping)
  double alpha_k = 0.0;

  /// Create a linear elastic material
  static Material linear_elastic(double youngs_modulus, double poissons_ratio,
                                 double density);

  /// Create a Neo-Hookean hyperelastic material
  static Material neo_hookean(double youngs_modulus, double poissons_ratio,
                              double density);

  /// Set Rayleigh damping coefficients
  Material with_damping(double alpha_m, double alpha_k) const;

  /// Get the 6x6 elasticity tensor (Voigt notation) for linear elasticity
  std::array<std::array<double, 6>, 6> elasticity_tensor() const;
};

class SolidMechanicsSolver {
  void assembleMassMatrix();
  void assembleStiffnessMatrix();
  void computeStrain();
  void computeStress();
  void computeVonMises();
  void computePrincipleStresses();

  bool is_initialized_;

  Material material_;
  std::shared_ptr<Mesh> mesh_ = nullptr;

  // SparseMatrices
  SparseMatrix mass_matrix_;
  SparseMatrix stiffness_matrix_;
  SparseMatrix damping_matrix_;

  // linear solver
  LinearSolver linear_solver_;

  // Primary fields
  std::vector<glm::dvec3>
      displacement_; // Incremental displacement (Updated Lagrangian)
  std::vector<glm::dvec3> velocity_;
  std::vector<glm::dvec3> acceleration_;

  // Previous time step values (for dynamics)
  std::vector<glm::dvec3> displacement_prev_;
  std::vector<glm::dvec3> velocity_prev;
  std::vector<glm::dvec3> acceleration_prev_;

  // Tracking fields for visualization
  std::vector<glm::dvec3> total_displacement_;
  std::vector<glm::dvec3> original_positions_;

  // derived quantities
  std::vector<glm::dmat3x3> strain_tensor_;
  std::vector<glm::dmat3x3> stress_tensor_;
  std::vector<double> von_mises_stress_;
  std::vector<glm::dvec3> principle_stresses_;

  // External loads
  std::vector<glm::dvec3> body_force_;
  std::vector<glm::dvec3> surface_traction_;

  // Boundary conditions
  std::vector<glm::dvec3> prescribed_displacements_;

  // FSI couplling
  std::vector<glm::dvec3> fsi_traction_;
  std::vector<size_t> fsi_nodes;

  // Solver settings
  size_t max_iter_;
  double tolerance_;
  double dt_;
  double time_;

  // Newmark-beta parameters for dynamics
  double beta_;
  double gamma_;
  size_t newton_max_iter_;
  double newton_tolerance_;

  // FSI coupling stabilization parameters
  double displacement_relaxation_;       // Under-relaxation factor (0 < α ≤ 1)
  size_t displacement_smoothing_iters_;  // Laplacian smoothing iterations
  double displacement_smoothing_factor_; // Smoothing weight (0 < λ < 1)
  double
      max_displacement_factor_; // Max displacement as fraction of element size

public:
  explicit SolidMechanicsSolver();

  /// Initialize solver with mesh and material
  void initialize(std::shared_ptr<Mesh> mesh_ptr, const Material &mat);

  /// Set fixed BC for nodes matching a condition
  void setFixedNodes(std::function<bool(glm::dvec3)> condition);

  /// Apply gravity body force
  void applyGravity(glm::dvec3 g = glm::dvec3{-9.8, 0.0, 0.0});

  /// Apply uniform pressure to boundary faces
  void applyPressure(double pressure);

  /// apply_pressure_per_node
  void applyPressure(const std::vector<double> &pressure);

  /// Add body force to a specific node
  void addBodyForce(size_t node_id, const glm::dvec3 &force);

  /// Solve static equilibrium: K * u = f
  bool solveStatic();

  /// Solve dynamic problem using Newmark-beta integration
  bool solveDynamicStep();

  /// Set FSI traction from fluid solver (for multiphysics coupling)
  void setFsiTraction(const std::vector<glm::dvec3> &traction);

  /// Get total displacement from original configuration (for visualization)
  const std::vector<glm::dvec3> &getTotalDisplacement() const;

  /// rebuild sparse matricies
  void rebuildSparseMatrices() {
    assembleMassMatrix();
    assembleStiffnessMatrix();
    // TODO: damping matrix
  }

  /// Get velocity field for FSI coupling
  const std::vector<glm::dvec3> &getVlocity() const;

  /// Update mesh positions based on displacement (for ALE/moving mesh)
  void deformMesh();

  /// Smooth displacement field using Laplacian smoothing
  void smoothDisplacementField();

  /// Get statistics for monitoring (max_disp, max_vm, max_vel)
  std::tuple<double, double, double> get_stats() const;

  // Accessors
  bool isSolverInitialized() const { return is_initialized_; }
  std::shared_ptr<Mesh> getMeshPtr() const { return mesh_; }
  const Material &getMaterial() const { return material_; }
  const std::vector<double> &getVonMisesStress() const {
    return von_mises_stress_;
  }
  const std::vector<glm::dmat3x3> &getStrain() const { return strain_tensor_; }
  const std::vector<glm::dmat3x3> &getStress() const { return stress_tensor_; }
  const std::vector<glm::dvec3> &getAcceleration() const {
    return acceleration_;
  }

  // Setters
  void setDt(double new_dt) { dt_ = new_dt; }
  void setMaxIter(size_t iter) { max_iter_ = iter; }
  void setTolerance(double tol) { tolerance_ = tol; }
  void setNewmarkParams(double new_beta, double new_gamma) {
    beta_ = new_beta;
    gamma_ = new_gamma;
  }

  void setMaterial(Material new_material) { material_ = new_material; }
  void setMeshPtr(std::shared_ptr<Mesh> new_mesh_ptr) { mesh_ = new_mesh_ptr; }

  double getDt() const { return dt_; }
  double getTime() const { return time_; }
};
