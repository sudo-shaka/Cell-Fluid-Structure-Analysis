#include <FEM/SolidMechanics.hpp>
#include <Mesh/Mesh.hpp>
#include <memory>

int main() {

  const auto mesh = Mesh::structuredRectangularPrism(20, 5, 40, 10, 10);
  const auto material = Material::linear_elastic(100, 1, 1000);
  auto solid_solver = SolidMechanicsSolver();
  solid_solver.initialize(std::make_shared<Mesh>(mesh), material);
  return 1;
}
