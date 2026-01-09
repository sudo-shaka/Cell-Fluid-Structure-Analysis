cmake_minimum_required(VERSION 3.10)
project(CellFluidStructureAnalysis LANGUAGES CXX)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags per configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(VERSION_DEBUG)
  add_compile_options(-g -O0 -Wall -Wextra -Wpedantic)
else()
  add_compile_definitions(VERSION_RELEASE)
  add_compile_options(-O3 -DNDEBUG -Wall)
endif()

## Build core library from src/DPM (and extendable later)
add_library(dpm STATIC
  src/DPM/DeformableParticle.cpp
)
target_include_directories(dpm PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Try to find GLM (header-only). If found, add include dirs for library and examples
find_package(glm QUIET)
if(glm_FOUND)
  target_include_directories(dpm PUBLIC ${GLM_INCLUDE_DIRS})
endif()

# Create one executable per example in the examples/ folder.
file(GLOB EXAMPLE_SRCS "${CMAKE_SOURCE_DIR}/examples/*.cpp")
foreach(example_src ${EXAMPLE_SRCS})
  get_filename_component(example_name ${example_src} NAME_WE)
  add_executable(${example_name} ${example_src})
  target_link_libraries(${example_name} PRIVATE dpm)
  target_include_directories(${example_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
endforeach()