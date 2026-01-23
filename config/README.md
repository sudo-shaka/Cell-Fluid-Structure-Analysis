# Simulation Configuration Files

This directory contains YAML configuration files for running different types of simulations.

## Overview

Instead of using hardcoded examples, the simulation now uses YAML files to configure:
- **Mesh**: Geometry (OBJ file, cylinder, or icosphere)
- **Navier-Stokes**: Fluid solver (optional)
- **Solid Mechanics**: Structural solver (optional)
- **DPM**: Deformable Particle Model for cells (optional)
- **FSI Coupling**: Fluid-Structure Interaction parameters (optional)
- **Output**: VTK export settings

## Available Configurations

### 1. `fsi_coupled.yaml`
**Fluid-Structure Interaction (FSI) coupling example**

Simulates a flexible tube/channel with fluid flow. Replicates the `FSICoupling.cpp` example.

- **Enabled solvers**: Navier-Stokes + Solid Mechanics
- **Coupling**: Explicit FSI coupling
- **Use case**: Flexible vessel with blood-like fluid flow

```bash
./simulation config/fsi_coupled.yaml
```

### 2. `fully_coupled.yaml`
**Fully coupled simulation (Fluid + Solid + DPM)**

Simulates blood vessel with fluid flow, wall mechanics, and deformable cells. Replicates the `CoupledSim.cpp` example.

- **Enabled solvers**: Navier-Stokes + Solid Mechanics + DPM
- **Use case**: Blood vessel with red blood cells

```bash
./simulation config/fully_coupled.yaml
```

### 3. `navier_stokes_only.yaml`
**Fluid-only simulation**

Runs only the Navier-Stokes solver without structural or particle coupling.

- **Enabled solvers**: Navier-Stokes only
- **Use case**: Pure fluid flow analysis

```bash
./simulation config/navier_stokes_only.yaml
```

### 4. `solid_mechanics_only.yaml`
**Structural mechanics-only simulation**

Runs only the solid mechanics solver without fluid or particle coupling.

- **Enabled solvers**: Solid Mechanics only
- **Use case**: Structural deformation analysis

```bash
./simulation config/solid_mechanics_only.yaml
```

### 5. `dpm_only.yaml`
**DPM-only simulation**

Runs only the deformable particle model without fluid or structural coupling.

- **Enabled solvers**: DPM only
- **Use case**: Cell dynamics and interactions

```bash
./simulation config/dpm_only.yaml
```

## Configuration Structure

### Mesh Configuration

```yaml
mesh:
  type: cylinder  # Options: obj, cylinder, icosphere
  
  # For OBJ type:
  # obj_file: /path/to/mesh.obj
  # obj_scale: 50.5
  
  # For cylinder type:
  length: 30.0
  radius: 2.0
  divisions: 35
  
  # For icosphere type:
  # radius: 1.0
  # recursion: 2
  
  max_element_size: 0.03
  flow_direction: [1.0, 0.0, 0.0]
  inlet_outlet_coverage: 1.0
```

### Solver Configurations

Each solver section has an `enabled` flag. If `enabled: false` or the section is empty, that solver won't run.

**Navier-Stokes (Fluid)**:
```yaml
navier_stokes:
  enabled: true
  density: 1060.0
  viscosity: 0.004
  inlet_velocity: [0.1, 0.0, 0.0]
  outlet_pressure: 0.0
  dt: 0.001
```

**Solid Mechanics**:
```yaml
solid_mechanics:
  enabled: true
  youngs_modulus: 5.0e5
  poisson_ratio: 0.45
  density: 1100.0
  dt: 0.001
```

**DPM (Deformable Cells)**:
```yaml
dpm:
  enabled: true
  cell_radius: 1.0
  num_cells: 64
  Kv: 5.0  # Volume constraint
  Ka: 1.0  # Area constraint
  dt: 0.005
```

**FSI Coupling** (only for Fluid + Solid):
```yaml
fsi_coupling:
  enabled: true
  scheme: explicit  # or implicit
  max_iterations: 10
  tolerance: 1.0e-4
```

### Output Configuration

```yaml
output:
  prefix: output_name
  total_steps: 5000
  output_frequency: 20  # Output every N steps
  vtk_scale: 1.0
  output_fluid: true
  output_solid: true
  output_dpm: false
```

## Creating Custom Configurations

1. Copy an existing configuration file
2. Modify the parameters as needed
3. Enable/disable solvers by setting `enabled: true/false`
4. Run with: `./simulation config/your_config.yaml`

### Simulation Types

The simulation automatically determines what type to run based on enabled solvers:

- **Single solver**: Only one solver enabled
- **FSI Coupled**: Navier-Stokes + Solid Mechanics (with FSI coupling enabled)
- **Fully Coupled**: All three solvers enabled (uses CoupledSolver)
- **Partial Coupled**: Any other combination of multiple solvers

## Building and Running

```bash
# Build the project
cd build
cmake ..
make simulation

# Run with a config file
./simulation ../config/fsi_coupled.yaml
```

## Output

VTK files will be generated with the naming pattern:
```
<prefix>_<step>_<solver>.vtk
```

For example:
- `fsi_output_0_fluid.vtk`
- `fsi_output_0_solid.vtk`
- `coupled_output_5_cells.vtk`

These can be visualized using ParaView or similar VTK viewers.