#pragma once

enum class BoundaryType {
  Wall,
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
