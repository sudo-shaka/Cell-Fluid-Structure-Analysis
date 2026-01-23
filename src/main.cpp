#include "DPM/DeformableParticle.hpp"
#include "DPM/ParticleInteractions.hpp"
#include "FEM/NavierStokes.hpp"
#include "FEM/SolidMechanics.hpp"
#include "IO/VtkExport.hpp"
#include "IO/YamlConfig.hpp"
#include "Mesh/Mesh.hpp"
#include "Numerics/CoupledSolver.hpp"
#include "Numerics/DPMIntegrator.hpp"
#include "Numerics/FEMIntegrator.hpp"
#include <iostream>
#include <memory>

std::shared_ptr<Mesh> createMesh(const config::MeshConfig &mesh_config) {
  std::shared_ptr<Mesh> mesh_ptr;

  switch (mesh_config.type) {
  case config::MeshConfig::MeshType::FromOBJ:
    std::cout << "Loading mesh from OBJ file: " << mesh_config.obj_file_path
              << std::endl;
    mesh_ptr = std::make_shared<Mesh>(
        Mesh::fromObjFile(mesh_config.obj_file_path, mesh_config.obj_scale));
    break;

  case config::MeshConfig::MeshType::Cylinder:
    std::cout << "Creating cylindrical mesh (L=" << mesh_config.cylinder_length
              << ", R=" << mesh_config.cylinder_radius << ")" << std::endl;
    mesh_ptr = std::make_shared<Mesh>(
        Mesh::fromPolyhedron(Polyhedron::cylendar(mesh_config.cylinder_length,
                                                  mesh_config.cylinder_radius,
                                                  mesh_config.cylinder_divisions),
                             mesh_config.max_element_size));
    break;

  case config::MeshConfig::MeshType::IcoSphere:
    std::cout << "Creating icosphere mesh (R=" << mesh_config.icosphere_radius
              << ")" << std::endl;
    mesh_ptr = std::make_shared<Mesh>(Mesh::fromPolyhedron(
        Polyhedron::isosphere(mesh_config.icosphere_radius,
                              mesh_config.icosphere_recursion),
        mesh_config.max_element_size));
    break;
  }

  std::cout << "  Mesh created: " << mesh_ptr->nVertices() << " vertices, "
            << mesh_ptr->nTets() << " tets, " << mesh_ptr->nFaces() << " faces"
            << std::endl;

  // Set up boundary conditions
  Mesh::setupBoundaryConditions(mesh_config.flow_direction,
                                mesh_config.inlet_outlet_coverage, *mesh_ptr);
  mesh_ptr->setP2BoundariesFromP1Boundaries();

  return mesh_ptr;
}

/**
 * @brief Run simulation based on configuration
 */
void runSimulation(const config::SimulationConfig &config) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "  Starting Simulation" << std::endl;
  std::cout << "========================================\n" << std::endl;

  // Create mesh
  auto mesh_ptr = createMesh(config.mesh);

  // Initialize coupled solver
  auto solver = std::make_shared<CoupledSolver>(mesh_ptr);

  // Initialize DPM solver if enabled
  if (config.dpm.enabled) {
    std::cout << "\nInitializing DPM solver with " << config.dpm.num_cells 
              << " cells..." << std::endl;
    
    std::vector<DeformableParticle> cells;
    cells.reserve(config.dpm.num_cells);
    for (int i = 0; i < config.dpm.num_cells; i++) {
      cells.emplace_back(config.dpm.icosphere_recursion, config.dpm.cell_radius);
    }
    for (auto &c : cells) {
      c.setKa(config.dpm.Ka);
      c.setKb(config.dpm.Kb);
      c.setKv(config.dpm.Kv);
    }

    DeformableParticle::Ks = config.dpm.Ks;
    DeformableParticle::Kre = config.dpm.Kre;
    DeformableParticle::Kat = config.dpm.Kat;

    solver->initializeDPMSolver(cells);
    solver->setDPMSolverDt(config.dpm.dt);

    // Initial cell relaxation
    if (config.dpm.initial_relaxation_steps > 0) {
      std::cout << "Performing initial cell relaxation (" 
                << config.dpm.initial_relaxation_steps << " steps)..." << std::endl;
      for (int i = 0; i < config.dpm.initial_relaxation_steps; i++) {
        solver->dpmStep();
      }
    }
  }

  // Initialize Navier-Stokes solver if enabled
  if (config.navier_stokes.enabled) {
    std::cout << "\nInitializing Navier-Stokes solver..." << std::endl;
    
    Fluid fluid_props;
    fluid_props.density = config.navier_stokes.density;
    fluid_props.viscosity = config.navier_stokes.viscosity;
    fluid_props.turbuelence_model = config.navier_stokes.is_laminar
                                        ? TurbulenceModel::Laminar
                                        : TurbulenceModel::RANS;
    fluid_props.viscosity_model = config.navier_stokes.is_newtonian
                                      ? ViscosityModel::Newtonian
                                      : ViscosityModel::Carreau;

    solver->setFluid(fluid_props);
    solver->initializeNavierStokesSolver();
    solver->setInletVelocity(config.navier_stokes.inlet_velocity);
    solver->setOutletPressure(config.navier_stokes.outlet_pressure);
    solver->setFluidSolverDt(config.navier_stokes.dt);
  }

  // Initialize Solid Mechanics solver if enabled
  if (config.solid_mechanics.enabled) {
    std::cout << "\nInitializing Solid Mechanics solver..." << std::endl;
    
    Material material = Material::linear_elastic(
        config.solid_mechanics.youngs_modulus,
        config.solid_mechanics.poisson_ratio,
        config.solid_mechanics.density);
    material = material.with_damping(config.solid_mechanics.damping_alpha,
                                     config.solid_mechanics.damping_beta);

    solver->setMaterial(material);
    solver->initializeSolidMechanicsSolver();
    solver->setMechanicsDt(config.solid_mechanics.dt);
  }

  // Check that at least one solver is enabled
  if (!config.navier_stokes.enabled && !config.solid_mechanics.enabled && !config.dpm.enabled) {
    std::cerr << "ERROR: No solver enabled in configuration!" << std::endl;
    return;
  }

  // Run simulation
  std::cout << "\nRunning simulation..." << std::endl;
  std::cout << "  Total steps: " << config.output.total_steps << std::endl;
  std::cout << "  Output frequency: " << config.output.output_frequency << std::endl;

  int steps_per_output = config.output.output_frequency;
  int num_outputs = config.output.total_steps / steps_per_output;

  for (int i = 0; i < num_outputs; i++) {
    std::cout << "\n========== Output Step " << i << " / " << num_outputs 
              << " ==========" << std::endl;
    
    // Write output
    std::string file_ext = "_" + std::to_string(i) + ".vtk";
    
    if (config.output.output_solid && solver->getMechanicsSolver()) {
      io::exportToVtk(config.output.output_prefix + "_solid" + file_ext,
                     *solver->getMechanicsSolver(), config.output.vtk_scale);
    }
    if (config.output.output_fluid && solver->getFluidSolver()) {
      io::exportToVtk(config.output.output_prefix + "_fluid" + file_ext,
                     *solver->getFluidSolver(), config.output.vtk_scale);
    }
    if (config.output.output_dpm && solver->getDPMSolver()) {
      io::exportToVtk(config.output.output_prefix + "_cells" + file_ext,
                     *solver->getDPMSolver(), config.output.vtk_scale);
    }

    // Advance simulation
    for (int j = 0; j < steps_per_output; j++) {
      if (config.isFSICoupled() || config.isFullyCoupled()) {
        solver->integrateStep();  // Coupled integration
      } else {
        // Single solver or non-FSI coupled
        if (config.dpm.enabled) solver->dpmStep();
        if (config.navier_stokes.enabled) solver->fluidStep();
        if (config.solid_mechanics.enabled) solver->mechanicsStep();
      }
    }

    // Print statistics
    if (solver->getFluidSolver()) {
      const auto &velocity = solver->getFluidSolver()->getVelocity();
      double max_vel = 0.0;
      for (const auto &v : velocity) {
        max_vel = std::max(max_vel, v.norm());
      }
      std::cout << "  Max fluid velocity: " << max_vel << " m/s" << std::endl;
    }
    if (solver->getMechanicsSolver()) {
      auto [max_disp, max_vm, max_vel] = solver->getMechanicsSolver()->get_stats();
      std::cout << "  Max solid displacement: " << max_disp << " m" << std::endl;
      std::cout << "  Max von Mises stress: " << max_vm << " Pa" << std::endl;
    }
    if (solver->getDPMSolver()) {
      std::cout << "  Number of cells: " << solver->getDPMSolver()->nParticles() << std::endl;
    }
  }

  std::cout << "\n========================================" << std::endl;
  std::cout << "  Simulation completed successfully!" << std::endl;
  std::cout << "========================================" << std::endl;
}

/**
 * @brief Run fully coupled simulation (Fluid + Solid + DPM)
 */
void runFullyCoupled(const config::SimulationConfig &config) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "  Fully Coupled Simulation (Fluid + Solid + DPM)" << std::endl;
  std::cout << "========================================\n" << std::endl;

  // Create mesh
  auto mesh_ptr = createMesh(config.mesh);

  // Create cell array for DPM
  std::vector<DeformableParticle> cells;
  cells.reserve(config.dpm.num_cells);
  for (int i = 0; i < config.dpm.num_cells; i++) {
    cells.emplace_back(config.dpm.icosphere_recursion, config.dpm.cell_radius);
  }
  for (auto &c : cells) {
    c.setKa(config.dpm.Ka);
    c.setKb(config.dpm.Kb);
    c.setKv(config.dpm.Kv);
  }

  // Set static parameters
  DeformableParticle::Ks = config.dpm.Ks;
  DeformableParticle::Kre = config.dpm.Kre;
  DeformableParticle::Kat = config.dpm.Kat;

  // Initialize coupled solver
  auto solver = std::make_shared<CoupledSolver>(mesh_ptr);
  solver->initializeDPMSolver(cells);

  // Initial cell relaxation
  std::cout << "Performing initial cell relaxation..." << std::endl;
  solver->setDPMSolverDt(config.dpm.dt);
  for (int i = 0; i < config.dpm.initial_relaxation_steps; i++) {
    solver->dpmStep();
  }

  // Initialize fluid solver
  Fluid fluid_props;
  fluid_props.density = config.navier_stokes.density;
  fluid_props.viscosity = config.navier_stokes.viscosity;
  fluid_props.turbuelence_model = config.navier_stokes.is_laminar
                                      ? TurbulenceModel::Laminar
                                      : TurbulenceModel::RANS;
  fluid_props.viscosity_model = config.navier_stokes.is_newtonian
                                    ? ViscosityModel::Newtonian
                                    : ViscosityModel::Carreau;

  solver->setFluid(fluid_props);
  solver->initializeNavierStokesSolver();
  solver->setInletVelocity(config.navier_stokes.inlet_velocity);
  solver->setOutletPressure(config.navier_stokes.outlet_pressure);

  // Initialize solid mechanics solver
  Material material = Material::linear_elastic(
      config.solid_mechanics.youngs_modulus,
      config.solid_mechanics.poisson_ratio, config.solid_mechanics.density);
  material = material.with_damping(config.solid_mechanics.damping_alpha,
                                   config.solid_mechanics.damping_beta);

  solver->setMaterial(material);
  solver->initializeSolidMechanicsSolver();

  // Set time steps
  solver->setDPMSolverDt(config.dpm.dt);
  solver->setFluidSolverDt(config.navier_stokes.dt);
  solver->setMechanicsDt(config.solid_mechanics.dt);

  // Run simulation
  std::cout << "\nRunning fully coupled simulation..." << std::endl;
  int steps_per_output = config.output.output_frequency;

  for (int i = 0; i < config.output.total_steps / steps_per_output; i++) {
    std::cout << "\n========  Output Step " << i << " ========" << std::endl;
    
    std::string file_ext = "_" + std::to_string(i) + ".vtk";
    
    if (config.output.output_solid && solver->getMechanicsSolver()) {
      io::exportToVtk(config.output.output_prefix + "_solid" + file_ext,
                     *solver->getMechanicsSolver(), config.output.vtk_scale);
    }
    if (config.output.output_fluid && solver->getFluidSolver()) {
      io::exportToVtk(config.output.output_prefix + "_fluid" + file_ext,
                     *solver->getFluidSolver(), config.output.vtk_scale);
    }
    if (config.output.output_dpm && solver->getDPMSolver()) {
      io::exportToVtk(config.output.output_prefix + "_cells" + file_ext,
                     *solver->getDPMSolver(), config.output.vtk_scale);
    }

    for (int j = 0; j < steps_per_output; j++) {
      solver->integrateStep();
    }
  }

  std::cout << "\n========================================" << std::endl;
  std::cout << "  Simulation completed successfully!" << std::endl;
  std::cout << "========================================" << std::endl;
}

/**
 * @brief Run Navier-Stokes only simulation
 */
void runNavierStokesOnly(const config::SimulationConfig &config) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "  Navier-Stokes Only Simulation" << std::endl;
  std::cout << "========================================\n" << std::endl;

  // Create mesh
  auto mesh_ptr = createMesh(config.mesh);

  // Initialize Navier-Stokes solver
  std::cout << "\nInitializing Navier-Stokes solver..." << std::endl;
  auto fluid_solver = std::make_shared<NavierStokesSolver>();

  Fluid fluid_props;
  fluid_props.density = config.navier_stokes.density;
  fluid_props.viscosity = config.navier_stokes.viscosity;
  fluid_props.turbuelence_model = config.navier_stokes.is_laminar
                                      ? TurbulenceModel::Laminar
                                      : TurbulenceModel::RANS;
  fluid_props.viscosity_model = config.navier_stokes.is_newtonian
                                    ? ViscosityModel::Newtonian
                                    : ViscosityModel::Carreau;

  fluid_solver->initialize(mesh_ptr, fluid_props);
  fluid_solver->setMeanInletVelocity(config.navier_stokes.inlet_velocity);
  
  if (config.navier_stokes.use_dirichlet_pressure) {
    fluid_solver->setOutletType(OutletType::DirichletPressure);
  }
  fluid_solver->setOutletPressure(config.navier_stokes.outlet_pressure);
  fluid_solver->setRelaxP(config.navier_stokes.relax_p);
  fluid_solver->setRelaxU(config.navier_stokes.relax_u);
  fluid_solver->setDt(config.navier_stokes.dt);

  // Create integrator
  auto integrator = std::make_shared<NavierStokesIntegrator>(fluid_solver);

  // Run simulation
  std::cout << "\nRunning Navier-Stokes simulation..." << std::endl;
  for (int step = 0; step < config.output.total_steps; ++step) {
    std::cout << "\n========== Time Step " << (step + 1) << " / "
              << config.output.total_steps << " ==========" << std::endl;
    std::cout << "Time: " << integrator->getTime() << " s" << std::endl;

    integrator->step();

    // Output results
    if (step % config.output.output_frequency == 0 ||
        step == config.output.total_steps - 1) {
      std::string filename = config.output.output_prefix + "_" +
                             std::to_string(step / config.output.output_frequency) + ".vtk";

      io::exportToVtk(filename, *fluid_solver, config.output.vtk_scale);
      std::cout << "  Output written: " << filename << std::endl;
    }

    // Print statistics
    const auto &velocity = fluid_solver->getVelocity();
    const auto &pressure = fluid_solver->getPressure();
    double max_vel = 0.0;
    double max_p = 0.0;
    for (const auto &v : velocity) {
      max_vel = std::max(max_vel, v.norm());
    }
    for (const auto &p : pressure) {
      max_p = std::max(max_p, std::abs(p));
    }

    std::cout << "  Max velocity: " << max_vel << " m/s" << std::endl;
    std::cout << "  Max pressure: " << max_p << " Pa" << std::endl;
  }

  std::cout << "\n========================================" << std::endl;
  std::cout << "  Simulation completed successfully!" << std::endl;
  std::cout << "========================================" << std::endl;
}

/**
 * @brief Run Solid Mechanics only simulation
 */
void runSolidMechanicsOnly(const config::SimulationConfig &config) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "  Solid Mechanics Only Simulation" << std::endl;
  std::cout << "========================================\n" << std::endl;

  // Create mesh
  auto mesh_ptr = createMesh(config.mesh);

  // Initialize Solid Mechanics solver
  std::cout << "\nInitializing Solid Mechanics solver..." << std::endl;
  auto solid_solver = std::make_shared<SolidMechanicsSolver>();

  Material material = Material::linear_elastic(
      config.solid_mechanics.youngs_modulus,
      config.solid_mechanics.poisson_ratio, 
      config.solid_mechanics.density);
  material = material.with_damping(config.solid_mechanics.damping_alpha,
                                   config.solid_mechanics.damping_beta);

  solid_solver->initialize(mesh_ptr, material);
  solid_solver->setDt(config.solid_mechanics.dt);
  solid_solver->setNewmarkParams(config.solid_mechanics.newmark_beta,
                                 config.solid_mechanics.newmark_gamma);

  // Create integrator
  auto integrator = std::make_shared<SolidMechanicsIntegrator>(solid_solver);

  // Run simulation
  std::cout << "\nRunning Solid Mechanics simulation..." << std::endl;
  for (int step = 0; step < config.output.total_steps; ++step) {
    std::cout << "\n========== Time Step " << (step + 1) << " / "
              << config.output.total_steps << " ==========" << std::endl;
    std::cout << "Time: " << integrator->getTime() << " s" << std::endl;

    integrator->step();

    // Output results
    if (step % config.output.output_frequency == 0 ||
        step == config.output.total_steps - 1) {
      std::string filename = config.output.output_prefix + "_" +
                             std::to_string(step / config.output.output_frequency) + ".vtk";

      io::exportToVtk(filename, *solid_solver, config.output.vtk_scale);
      std::cout << "  Output written: " << filename << std::endl;
    }

    // Print statistics
    auto [max_disp, max_vm, max_vel] = solid_solver->get_stats();
    std::cout << "  Max displacement: " << max_disp << " m" << std::endl;
    std::cout << "  Max von Mises stress: " << max_vm << " Pa" << std::endl;
    std::cout << "  Max velocity: " << max_vel << " m/s" << std::endl;
  }

  std::cout << "\n========================================" << std::endl;
  std::cout << "  Simulation completed successfully!" << std::endl;
  std::cout << "========================================" << std::endl;
}

/**
 * @brief Run DPM only simulation
 */
void runDPMOnly(const config::SimulationConfig &config) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "  DPM Only Simulation" << std::endl;
  std::cout << "========================================\n" << std::endl;

  // Create cell array for DPM with random initial positions
  std::cout << "\nCreating " << config.dpm.num_cells << " deformable cells..." << std::endl;
  std::vector<DeformableParticle> cells;
  cells.reserve(config.dpm.num_cells);
  
  // Seed random number generator
  std::srand(42);
  
  // Create cells at random positions to avoid all being at (0,0,0)
  double spacing = config.dpm.cell_radius * 3.0;
  for (int i = 0; i < config.dpm.num_cells; i++) {
    Eigen::Vector3d start_pos(
      (std::rand() / (double)RAND_MAX - 0.5) * spacing * 5,
      (std::rand() / (double)RAND_MAX - 0.5) * spacing * 5,
      (std::rand() / (double)RAND_MAX - 0.5) * spacing * 5
    );
    cells.emplace_back(start_pos, 1.0, config.dpm.icosphere_recursion, 
                      config.dpm.cell_radius, config.dpm.Kv, config.dpm.Ka, config.dpm.Kb);
  }

  // Set static parameters
  DeformableParticle::Ks = config.dpm.Ks;
  DeformableParticle::Kre = config.dpm.Kre;
  DeformableParticle::Kat = config.dpm.Kat;

  // Initialize DPM solver
  auto dpm_solver = std::make_shared<ParticleInteractions>(cells);
  auto integrator = std::make_shared<DPMTimeIntegrator>(dpm_solver);
  integrator->setDT(config.dpm.dt);

  // Run simulation
  std::cout << "\nRunning DPM simulation..." << std::endl;
  double time = 0.0;
  
  for (int step = 0; step < config.output.total_steps; ++step) {
    std::cout << "\n========== Time Step " << (step + 1) << " / "
              << config.output.total_steps << " ==========" << std::endl;
    std::cout << "Time: " << time << " s" << std::endl;

    integrator->advanceStep();
    time += config.dpm.dt;

    // Output results
    if (step % config.output.output_frequency == 0 ||
        step == config.output.total_steps - 1) {
      std::string filename = config.output.output_prefix + "_" +
                             std::to_string(step / config.output.output_frequency) + ".vtk";

      io::exportToVtk(filename, *dpm_solver, config.output.vtk_scale);
      std::cout << "  Output written: " << filename << std::endl;
    }

    // Print statistics
    std::cout << "  Number of cells: " << dpm_solver->nParticles() << std::endl;
    std::cout << "  Simulation time: " << time << " s" << std::endl;
  }

  std::cout << "\n========================================" << std::endl;
  std::cout << "  Simulation completed successfully!" << std::endl;
  std::cout << "========================================" << std::endl;
}

/**
 * @brief Run single solver simulation
 */
void runSingleSolver(const config::SimulationConfig &config) {
  if (config.navier_stokes.enabled) {
    runNavierStokesOnly(config);
  } else if (config.solid_mechanics.enabled) {
    runSolidMechanicsOnly(config);
  } else if (config.dpm.enabled) {
    runDPMOnly(config);
  } else {
    std::cerr << "ERROR: No solver enabled in configuration!" << std::endl;
  }
}

int main(int argc, char *argv[]) {
  std::cout << "========================================" << std::endl;
  std::cout << "  Cell-Fluid-Structure Analysis" << std::endl;
  std::cout << "  YAML Configuration-Based Simulation" << std::endl;
  std::cout << "========================================\n" << std::endl;

  // Parse command-line arguments
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <config.yaml>" << std::endl;
    std::cerr << "\nExample configurations can be found in the config/ "
                 "directory."
              << std::endl;
    return 1;
  }

  std::string config_file = argv[1];

  try {
    // Load configuration
    auto config = config::loadConfig(config_file);

    // Run simulation
    runSimulation(config);

  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
