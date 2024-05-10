cmake_minimum_required(VERSION 3.18)
project(DerivedPerf C CXX)

## Options:
##   BUILD_KERNELS - code for simple kernels using Kokkos
##   BUILD_WRITE_BENCH - codes to benchmark read/write using ADIOS2 and Kokkos
##   BUILD_ADIOS_DERIVE - code for testing the performance of derived functions in ADIOS2

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (BUILD_KERNELS OR BUILD_WRITE_BENCH)
  find_package(Kokkos REQUIRED)
  if(DEFINED Kokkos_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${Kokkos_CXX_COMPILER}")
    message("Compiler used: ${Kokkos_CXX_COMPILER}")
  endif()
endif()

if (BUILD_ADIOS_DERIVE OR BUILD_WRITE_BENCH)
  find_package(adios2 REQUIRED)
endif()

if (BUILD_WRITE_BENCH)
  find_package(MPI REQUIRED)
endif()

if (BUILD_WRITE_BENCH)
    ## No tests yet
endif()

if (BUILD_ADIOS_DERIVE)
    add_executable(perf_adios adiosDerivedPerf.cpp)
    target_link_libraries(perf_adios adios2::adios2 adios2::cxx11)
endif()

if (BUILD_KERNELS)
    add_executable(perf_kernel kernelDerivedPerf.cpp)
    kokkos_compilation(SOURCE kernelDerivedPerf.cpp)
    target_link_libraries(perf_kernel Kokkos::kokkos)
endif()
