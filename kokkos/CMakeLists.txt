cmake_minimum_required(VERSION 3.15)
project(miniMD)

find_package(Kokkos REQUIRED)

OPTION(MINIMD_ENABLE_MPI "" OFF)





add_executable(miniMD 
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
target_link_libraries(miniMD Kokkos::kokkos)


if(MINIMD_ENABLE_MPI)
else()
  add_library(mpi_stubs STATIC MPI-Stubs/mpi.c)
  set_target_properties(mpi_stubs PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(mpi_stubs INTERFACE MPI-Stubs)
  target_link_libraries(miniMD mpi_stubs)
endif()

enable_testing()
file(RELATIVE_PATH REL_BINARY_DIR ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME simple COMMAND bash run_tests 0 lj 0 "${REL_BINARY_DIR}/"
         WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})