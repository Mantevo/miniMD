# Kokkos minimally requires 3.16 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.16)

# Projects can safely mix languages - must have C++ support
# Kokkos flags will only apply to C++ files
project(Example CXX C)

# Look for an installed Kokkos
find_package(Kokkos REQUIRED)

set(SOURCE_FILES
  atom.cpp
  comm.cpp
  force_eam.cpp
  force_lj.cpp
  input.cpp
  integrate.cpp
  ljs.cpp
  neighbor.cpp
  output.cpp
  setup.cpp
  thermo.cpp
  timer.cpp
)

if(MPI_STUBS)
  list(APPEND SOURCE_FILES MPI-Stubs/mpi.c)
  include_directories(MPI-Stubs)
endif()

add_executable(miniMD ${SOURCE_FILES})

# This is the only thing required to set up compiler/linker flags
target_link_libraries(miniMD Kokkos::kokkos)

