#include "IO/YamlConfig.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <stdexcept>

namespace config {

namespace {

// Helper to read optional values
template<typename T>
T getOr(const YAML::Node& node, const std::string& key, T default_value) {
  if (node[key]) {
    return node[key].as<T>();
  }
  return default_value;
}

Eigen::Vector3d parseVector3d(const YAML::Node& node) {
  if (node.IsSequence() && node.size() == 3) {
    return Eigen::Vector3d(
      node[0].as<double>(),
      node[1].as<double>(),
      node[2].as<double>()
    );
  }
  throw std::runtime_error("Invalid Vector3d format - expected [x, y, z]");
}

MeshConfig parseMeshConfig(const YAML::Node& node) {
  MeshConfig config;

  if (!node["type"]) {
    throw std::runtime_error("Mesh type not specified");
  }

  std::string type_str = node["type"].as<std::string>();
  if (type_str == "obj") {
    config.type = MeshConfig::MeshType::FromOBJ;
    if (!node["obj_file"]) {
      throw std::runtime_error("obj_file required for OBJ mesh type");
    }
    config.obj_file_path = node["obj_file"].as<std::string>();
    config.obj_scale = getOr(node, "obj_scale", 1.0);
  } else if (type_str == "cylinder") {
    config.type = MeshConfig::MeshType::Cylinder;
    config.cylinder_length = getOr(node, "length", 30.0);
    config.cylinder_radius = getOr(node, "radius", 2.0);
    config.cylinder_divisions = getOr(node, "divisions", 35);
  } else if (type_str == "icosphere") {
    config.type = MeshConfig::MeshType::IcoSphere;
    config.icosphere_radius = getOr(node, "radius", 1.0);
    config.icosphere_recursion = getOr(node, "recursion", 2);
  } else {
    throw std::runtime_error("Unknown mesh type: " + type_str);
  }

  config.max_element_size = getOr(node, "max_element_size", 0.03);

  if (node["flow_direction"]) {
    config.flow_direction = parseVector3d(node["flow_direction"]);
  }
  config.inlet_outlet_coverage = getOr(node, "inlet_outlet_coverage", 1.0);

  return config;
}

NavierStokesConfig parseNavierStokesConfig(const YAML::Node& node) {
  NavierStokesConfig config;

  if (!node || !node.IsMap()) {
    return config; // Return with enabled=false
  }

  config.enabled = getOr(node, "enabled", false);
  if (!config.enabled) {
    return config;
  }

  config.density = getOr(node, "density", 1060.0);
  config.viscosity = getOr(node, "viscosity", 0.004);
  config.is_newtonian = getOr(node, "is_newtonian", true);
  config.is_laminar = getOr(node, "is_laminar", true);

  if (node["inlet_velocity"]) {
    config.inlet_velocity = parseVector3d(node["inlet_velocity"]);
  }
  config.outlet_pressure = getOr(node, "outlet_pressure", 0.0);
  config.use_dirichlet_pressure = getOr(node, "use_dirichlet_pressure", true);

  config.dt = getOr(node, "dt", 1e-3);
  config.relax_p = getOr(node, "relax_p", 0.5);
  config.relax_u = getOr(node, "relax_u", 0.5);

  return config;
}

SolidMechanicsConfig parseSolidMechanicsConfig(const YAML::Node& node) {
  SolidMechanicsConfig config;

  if (!node || !node.IsMap()) {
    return config; // Return with enabled=false
  }

  config.enabled = getOr(node, "enabled", false);
  if (!config.enabled) {
    return config;
  }

  config.youngs_modulus = getOr(node, "youngs_modulus", 5e5);
  config.poisson_ratio = getOr(node, "poisson_ratio", 0.45);
  config.density = getOr(node, "density", 1100.0);

  config.damping_alpha = getOr(node, "damping_alpha", 0.01);
  config.damping_beta = getOr(node, "damping_beta", 0.001);

  config.dt = getOr(node, "dt", 1e-3);
  config.newmark_beta = getOr(node, "newmark_beta", 0.25);
  config.newmark_gamma = getOr(node, "newmark_gamma", 0.5);

  return config;
}

DPMConfig parseDPMConfig(const YAML::Node& node) {
  DPMConfig config;

  if (!node || !node.IsMap()) {
    return config; // Return with enabled=false
  }

  config.enabled = getOr(node, "enabled", false);
  if (!config.enabled) {
    return config;
  }

  config.cell_radius = getOr(node, "cell_radius", 1.0);
  config.icosphere_recursion = getOr(node, "icosphere_recursion", 2);
  config.num_cells = getOr(node, "num_cells", 64);

  config.Kv = getOr(node, "Kv", 5.0);
  config.Ka = getOr(node, "Ka", 1.0);
  config.Kb = getOr(node, "Kb", 0.0);
  config.Ks = getOr(node, "Ks", 10.0);
  config.Kre = getOr(node, "Kre", 20.0);
  config.Kat = getOr(node, "Kat", 0.3);

  config.dt = getOr(node, "dt", 0.005);
  config.initial_relaxation_steps = getOr(node, "initial_relaxation_steps", 250);

  return config;
}

FSICouplingConfig parseFSICouplingConfig(const YAML::Node& node) {
  FSICouplingConfig config;

  if (!node || !node.IsMap()) {
    return config; // Return with enabled=false
  }

  config.enabled = getOr(node, "enabled", false);
  if (!config.enabled) {
    return config;
  }

  std::string scheme_str = getOr<std::string>(node, "scheme", "explicit");
  if (scheme_str == "implicit") {
    config.scheme = FSICouplingConfig::CouplingScheme::Implicit;
  } else {
    config.scheme = FSICouplingConfig::CouplingScheme::Explicit;
  }

  config.max_iterations = getOr(node, "max_iterations", 10);
  config.tolerance = getOr(node, "tolerance", 1e-4);
  config.enable_mesh_update = getOr(node, "enable_mesh_update", true);
  config.enable_matrix_rebuild = getOr(node, "enable_matrix_rebuild", true);

  return config;
}

OutputConfig parseOutputConfig(const YAML::Node& node) {
  OutputConfig config;

  if (!node || !node.IsMap()) {
    return config; // Return defaults
  }

  config.output_prefix = getOr<std::string>(node, "prefix", "output");
  config.total_steps = getOr(node, "total_steps", 1000);
  config.output_frequency = getOr(node, "output_frequency", 20);
  config.vtk_scale = getOr(node, "vtk_scale", 1.0);
  config.output_fluid = getOr(node, "output_fluid", true);
  config.output_solid = getOr(node, "output_solid", true);
  config.output_dpm = getOr(node, "output_dpm", true);

  return config;
}

} // anonymous namespace

SimulationConfig loadConfig(const std::string& yaml_file) {
  try {
    YAML::Node root = YAML::LoadFile(yaml_file);
    
    SimulationConfig config;

    // Parse mesh config (required)
    if (!root["mesh"]) {
      throw std::runtime_error("Mesh configuration is required");
    }
    config.mesh = parseMeshConfig(root["mesh"]);

    // Parse solver configs (optional)
    config.navier_stokes = parseNavierStokesConfig(root["navier_stokes"]);
    config.solid_mechanics = parseSolidMechanicsConfig(root["solid_mechanics"]);
    config.dpm = parseDPMConfig(root["dpm"]);
    config.fsi_coupling = parseFSICouplingConfig(root["fsi_coupling"]);
    config.output = parseOutputConfig(root["output"]);

    std::cout << "Configuration loaded from: " << yaml_file << std::endl;
    std::cout << "  Navier-Stokes: " << (config.navier_stokes.enabled ? "enabled" : "disabled") << std::endl;
    std::cout << "  Solid Mechanics: " << (config.solid_mechanics.enabled ? "enabled" : "disabled") << std::endl;
    std::cout << "  DPM: " << (config.dpm.enabled ? "enabled" : "disabled") << std::endl;
    std::cout << "  FSI Coupling: " << (config.fsi_coupling.enabled ? "enabled" : "disabled") << std::endl;

    return config;

  } catch (const YAML::Exception& e) {
    throw std::runtime_error("YAML parsing error: " + std::string(e.what()));
  } catch (const std::exception& e) {
    throw std::runtime_error("Configuration error: " + std::string(e.what()));
  }
}

} // namespace config
