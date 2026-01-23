#pragma once

#include <Eigen/Core>
#include <string>

namespace config {

struct MeshConfig {
  enum class MeshType { FromOBJ, Cylinder, IcoSphere };

  MeshType type = MeshType::Cylinder;

  // For FromOBJ type
  std::string obj_file_path;
  double obj_scale = 1.0;

  // For Cylinder type
  double cylinder_length = 30.0;
  double cylinder_radius = 2.0;
  int cylinder_divisions = 35;

  // For IcoSphere type
  double icosphere_radius = 1.0;
  int icosphere_recursion = 2;

  // Mesh quality
  double max_element_size = 0.03;

  // Boundary setup
  Eigen::Vector3d flow_direction = Eigen::Vector3d(1.0, 0.0, 0.0);
  double inlet_outlet_coverage = 1.0;
};

/**
 * @brief Navier-Stokes fluid solver configuration
 */
struct NavierStokesConfig {
  bool enabled = false;

  // Fluid properties
  double density = 1060.0;     // kg/m³
  double viscosity = 0.004;    // Pa·s
  bool is_newtonian = true;
  bool is_laminar = true;

  // Inlet/Outlet conditions
  Eigen::Vector3d inlet_velocity = Eigen::Vector3d(0.1, 0.0, 0.0);
  double outlet_pressure = 0.0;
  bool use_dirichlet_pressure = true;

  // Solver parameters
  double dt = 1e-3;
  double relax_p = 0.5;
  double relax_u = 0.5;
};

/**
 * @brief Solid mechanics solver configuration
 */
struct SolidMechanicsConfig {
  bool enabled = false;

  // Material properties
  double youngs_modulus = 5e5;  // Pa
  double poisson_ratio = 0.45;
  double density = 1100.0;      // kg/m³

  // Damping parameters
  double damping_alpha = 0.01;
  double damping_beta = 0.001;

  // Time integration
  double dt = 1e-3;
  double newmark_beta = 0.25;
  double newmark_gamma = 0.5;
};

struct DPMConfig {
  bool enabled = false;

  // Cell properties
  double cell_radius = 1.0;
  int icosphere_recursion = 2;
  int num_cells = 64;

  // Spring constants
  double Kv = 5.0;   // Volume constraint
  double Ka = 1.0;   // Area constraint
  double Kb = 0.0;   // Bending
  double Ks = 10.0;  // Stretching
  double Kre = 20.0; // Repulsion
  double Kat = 0.3;  // Attraction

  // Time integration
  double dt = 0.005;
  int initial_relaxation_steps = 250;
};

/**
 * @brief FSI coupling configuration
 */
struct FSICouplingConfig {
  bool enabled = false;

  enum class CouplingScheme { Explicit, Implicit };
  CouplingScheme scheme = CouplingScheme::Explicit;

  int max_iterations = 10;
  double tolerance = 1e-4;
  bool enable_mesh_update = true;
  bool enable_matrix_rebuild = true;
};

/**
 * @brief Simulation output configuration
 */
struct OutputConfig {
  std::string output_prefix = "output";
  int total_steps = 1000;
  int output_frequency = 20;
  double vtk_scale = 1.0;  // Scale factor for VTK export
  bool output_fluid = true;
  bool output_solid = true;
  bool output_dpm = true;
};

/**
 * @brief Main simulation configuration
 */
struct SimulationConfig {
  MeshConfig mesh;
  NavierStokesConfig navier_stokes;
  SolidMechanicsConfig solid_mechanics;
  DPMConfig dpm;
  FSICouplingConfig fsi_coupling;
  OutputConfig output;

  // Determine what type of simulation to run
  bool isCoupledSimulation() const {
    int enabled_count = 0;
    if (navier_stokes.enabled) enabled_count++;
    if (solid_mechanics.enabled) enabled_count++;
    if (dpm.enabled) enabled_count++;
    return enabled_count > 1;
  }

  bool isFSICoupled() const {
    return navier_stokes.enabled && solid_mechanics.enabled && 
           !dpm.enabled && fsi_coupling.enabled;
  }

  bool isFullyCoupled() const {
    return navier_stokes.enabled && solid_mechanics.enabled && dpm.enabled;
  }
};

SimulationConfig loadConfig(const std::string& yaml_file);

} // namespace config
