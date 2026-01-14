#pragma once

#include <glm/vec3.hpp>

// forward declarations
class Mesh;
class VectorField;
class ScalarField;
class NavierStokesSolver;
class SolidMechanicsSolver;
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

enum class FluidBCType {
  Wall, // noSlip
  Inlet,
  Outlet,
  Internal,
  Undefined,
};

enum class OutletType {
  DirichletPressure,
  Neumann,
  Undefined,
};

enum class InletType {
  Uniform,
  Pulsitile,
  Undefined,
};

namespace boundary_assignment {
void setupBoundaryConditions(
    const glm::dvec3 &intlet_to_outlet_direction,
    Mesh &mesh, // mesh for boundaries to be assigned to.
    const double percent_inlet_to_outlet_converage = 5.0);
void applyBoundaryConditions(NavierStokesSolver &solver);
void applyBoundaryConditions(SolidMechanicsSolver &solver);
} // namespace boundary_assignment
