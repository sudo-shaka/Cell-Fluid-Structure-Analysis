# Cell-Fluid-Structure Analysis

A work-in-progress C++ simulation framework for coupled fluid-structure-cell interactions, with applications in biomedical engineering and computational biology. This project implements multi-physics simulations combining Navier-Stokes fluid dynamics, solid mechanics (FEM), and deformable particle models (DPM) for cellular dynamics.

![Simulation Example](tranverse_example.gif)
*Example: Vessel seeded with endothlial cells at the onset of flow*

## Features

- **Navier-Stokes Solver**: CFD simulation for incompressible fluid flow with support for laminar and turbulent models
- **Solid Mechanics**: Finite Element Method (FEM) for structural deformation analysis
- **Deformable Particle Model (DPM)**: Simulation of deformable cells and particles
- **Fluid-Structure Interaction (FSI)**: Coupled fluid-solid simulations
- **Fully Coupled Simulations**: Combined fluid, structure, and cell dynamics (e.g., blood flow with vessel wall and red blood cells)
- **Flexible Mesh Generation**: Support for cylinder, icosphere, and custom OBJ geometries
- **YAML Configuration**: Easy-to-use configuration files for different simulation scenarios
- **VTK Export**: Visualization-ready output for ParaView and other VTK-compatible tools

## Prerequisites

### Required Dependencies

- **C++17** compatible compiler (GCC, Clang, or MSVC)
- **CMake** 3.10 or higher
- **Eigen3** 3.3 or higher (for linear algebra and sparse solvers)
- **yaml-cpp** (for configuration file parsing)
- **TetGen** 1.6 (1.5 often in lib packages, clone 1.6 to wd if 1.6 is not system lib version) (for tetrahedral mesh generation)

### Installing Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install cmake libeigen3-dev libyaml-cpp-dev
```

#### macOS (Homebrew)
```bash
brew install cmake eigen yaml-cpp
```

#### TetGen
Clone TetGen into the project directory:
```bash
git clone https://github.com/TetGen/TetGen.git
```

## Building the Project

```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build (adjust -j flag based on your CPU cores)
make -j $(nproc)
```

### Build Types

- **Release** (default): Optimized for performance (`-O3`)
- **Debug**: Includes debugging symbols (`-g -O0`)

```bash
# For debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j $(nproc)
```

## Usage

### Configuration-Based Simulation (Recommended)

The primary way to run simulations is through YAML configuration files:

```bash
./simulation config/fully_coupled.yaml
```

### Available Configurations

The [config/](config/) directory contains pre-configured simulation setups:

| Configuration | Description | Solvers Enabled |
|--------------|-------------|-----------------|
| `fully_coupled.yaml` | Blood vessel with fluid, wall mechanics, and cells | Navier-Stokes + Solid + DPM |
| `fsi.yaml` | Fluid-structure interaction | Navier-Stokes + Solid |
| `navier_stokes_only.yaml` | Pure fluid flow simulation | Navier-Stokes |
| `solid_mechanics_only.yaml` | Structural deformation only | Solid Mechanics |
| `dpm_only.yaml` | Cell dynamics only | DPM |

See [config/README.md](config/README.md) for detailed configuration documentation.

### Example Programs

Standalone examples demonstrating individual components:

```bash
# Coupled blood vessel simulation
./CoupledSim

# Fluid flow in a vessel
./NavierStokes

# Structural mechanics
./StructuralDynamics

# Cell tissue simulation
./DPMTissue

# Mesh generation
./mesh
./cylendar
./icosphere
```

## Project Structure

```
Cell-Fluid-Structure-Analysis/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── config/                 # YAML configuration files
│   ├── README.md          # Configuration documentation
│   ├── fully_coupled.yaml
│   ├── fsi.yaml
│   └── ...
├── examples/               # Standalone example programs
│   ├── CoupledSim.cpp     # Fully coupled blood vessel
│   ├── NavierStokes.cpp   # Fluid flow example
│   ├── DPMTissue.cpp      # Cell tissue example
│   └── ...
├── include/                # Header files
│   ├── DPM/               # Deformable Particle Model
│   ├── FEM/               # Finite Element Method
│   ├── IO/                # Input/Output utilities
│   ├── Mesh/              # Mesh generation and handling
│   ├── Numerics/          # Numerical solvers
│   └── Polyhedron/        # Geometric primitives
├── src/                    # Implementation files
│   ├── main.cpp           # Configuration-based simulation entry
│   ├── DPM/
│   ├── FEM/
│   └── ...
├── meshes/                 # Mesh files (OBJ, etc.)
└── build/                  # Build outputs and VTK files
```

## Visualization

Simulation results are exported as VTK files in wording directory

- `coupled_output_fluid_*.vtk` - Fluid domain results
- `coupled_output_solid_*.vtk` - Structural domain results
- `coupled_output_cells_*.vtk` - Cell/particle results

3. Use the time controls to animate through timesteps

## Configuration

### Basic YAML Configuration Example

```yaml
mesh:
  type: cylinder
  length: 30.0
  radius: 2.0
  divisions: 35
  max_element_size: 0.03

navier_stokes:
  enabled: true
  density: 1060.0
  viscosity: 0.004
  inlet_velocity: 0.5
  timestep: 0.01
  num_steps: 100

solid_mechanics:
  enabled: true
  youngs_modulus: 60000.0
  poisson_ratio: 0.49
  density: 1060.0

output:
  vtk_enabled: true
  vtk_prefix: simulation_output
  vtk_scale: 50.0
```

See [config/README.md](config/README.md) for complete configuration options.

## Development

### Code Organization

- **DPM/**: Deformable Particle Model for cell simulations
  - Cell membrane mechanics
  - Inter-cell interactions
  - Volume and area constraints

- **FEM/**: Finite Element Method implementations
  - `NavierStokes.hpp`: Incompressible fluid solver
  - `SolidMechanics.hpp`: Linear/nonlinear elasticity
  - Material models and constitutive laws

- **Mesh/**: Mesh generation and boundary conditions
  - Tetrahedral mesh generation (via TetGen)
  - Surface and volume mesh handling
  - Boundary condition setup

- **Numerics/**: Numerical solvers and algorithms
  - `CoupledSolver.hpp`: Multi-physics coupling
  - Time integration schemes
  - Linear system solvers

- **IO/**: Input/Output utilities
  - VTK export for visualization
  - Configuration file parsing

## Examples and Use Cases

### Blood Flow in Vessels
Simulate blood flow through deformable vessel walls with red blood cells:
```bash
./simulation config/fully_coupled.yaml
```

### Soft Tissue Mechanics
Analyze structural deformation of biological tissues:
```bash
./simulation config/solid_mechanics_only.yaml
```

### Cell Dynamics
Model cellular interactions and tissue formation:
```bash
./simulation config/dpm_only.yaml
```

## Troubleshooting

### Common Issues

1. **TetGen not found**
   ```
   Solution: Clone TetGen into the project root directory
   git clone https://github.com/TetGen/TetGen.git
   ```

2. **Eigen3 not found**
   ```
   Solution: Install Eigen3 development headers
   sudo apt-get install libeigen3-dev
   ```

3. **yaml-cpp not found**
   ```
   Solution: Install yaml-cpp library
   sudo apt-get install libyaml-cpp-dev
   ```

4. **Build errors with C++ standard**
   ```
   Solution: Ensure your compiler supports C++17
   ```

## Contributing

Contributions are welcome! Areas for improvement:
- Additional turbulence models
- Non-Newtonian fluid models
- Advanced material models
- GPU acceleration
- Parallel processing support

## Citation

In preparation:
```
[Hopefully will be published...]
```
