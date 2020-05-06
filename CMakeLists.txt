cmake_minimum_required (VERSION 3.5.0)

project (Smurff)

MESSAGE("smurff")

# OPTIONS

SET(ALGEBRA_LIB_NAME "LAPACKE" CACHE STRING "BLAS/LaPACK Library")
SET_PROPERTY(CACHE ALGEBRA_LIB_NAME PROPERTY STRINGS MKL LAPACKE OPENBLAS)

SET(ACCEL_BACKEND "None" CACHE STRING "Acceleration Backend Library")
SET_PROPERTY(CACHE ACCEL_BACKEND PROPERTY STRINGS None CUDA OpenCL)

SET(ACCEL_FRONTEND "None" CACHE STRING "Acceleration Frontend Library")
SET_PROPERTY(CACHE ACCEL_FRONTEND PROPERTY STRINGS None ViennaCL ArrayFire)

OPTION(ENABLE_PROFILING "Enable performance counters" OFF)
OPTION(ENABLE_ASAN "Enable address sanitizer in Debug mode" OFF)

OPTION(ENABLE_BOOST "Enable usage of Boost for program options" ON)

OPTION(ENABLE_PYTHON "Enable Python Wrapper" ON)
OPTION(ENABLE_MPI "Enable MPI Support" OFF)

SET(SMURFF_FLOAT_TYPE "double" CACHE STRING "Main compute type")

# INIT CMAKE

message("Initializing cmake ...")
MESSAGE( STATUS "CMAKE_VERSION: " ${CMAKE_VERSION} )
# the short system name, e.g. "Linux", "FreeBSD" or "Windows"
MESSAGE( STATUS "CMAKE_SYSTEM_NAME: " ${CMAKE_SYSTEM_NAME} )
# only the version part of CMAKE_SYSTEM
MESSAGE( STATUS "CMAKE_SYSTEM_VERSION: " ${CMAKE_SYSTEM_VERSION} )

# Add new build type: Release w/o OpenMP (RelNoOpenMP)
SET(CMAKE_CXX_FLAGS_RELNOOPENMP "${GCC_RELEASE_FLAGS}" CACHE STRING "Flags used by the C++ compiler for release-without-OpenMP builds.")
SET(CMAKE_C_FLAGS_RELNOOPENMP "${GCC_RELEASE_FLAGS}" CACHE STRING "Flags used by the C compiler for release-without-OpenMP builds.")
SET(CMAKE_EXE_LINKER_FLAGS_RELNOOPENMP "" CACHE STRING "Flags used for linking binaries for release-without-OpenMP builds.")
SET(CMAKE_SHARED_LINKER_FLAGS_RELNOOPENMP "" CACHE STRING "Flags used by the shared libraries for release-without-OpenMP builds.")
MARK_AS_ADVANCED( CMAKE_CXX_FLAGS_RELNOOPENMP CMAKE_C_FLAGS_RELNOOPENMP CMAKE_EXE_LINKER_FLAGS_RELNOOPENMP CMAKE_SHARED_LINKER_FLAGS_RELNOOPENMP )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# needed for PyBIND11
# https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

if (NOT MSVC)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wnon-virtual-dtor -Wno-parentheses")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unknown-pragmas")
   set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -fstrict-aliasing")
   set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g -fstrict-aliasing")

   if (${ENABLE_ASAN})
       set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
       set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
   endif()
endif()

include(cmake/InitGit.cmake)
init_git()

# CONFIGURE BUILD TYPE

message("Analyzing build type...")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

if(NOT MSVC)
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Debug build.")
    add_definitions(-D_DEBUG)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
  elseif(CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Release build.")
  else()
    message(STATUS "Unknown build type...")
  endif()
endif()

# CONFIGURE DEPENDENCIES

message("Checking build dependencies...")

#set additional search paths for cmake scripts
set (SCRIPT_DIR "${CMAKE_SOURCE_DIR}/cmake/")
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
include(cmake/DependenciesConfig.cmake)

configure_highfive()
configure_boost()
configure_eigen()
configure_openmp()

# MKL, OpenBLAS, ...
include(cmake/ConfigureAlgebra.cmake)
configure_algebra()

include(cmake/ConfigureAccel.cmake)
report_accel()
configure_accel_backend() # OpenCL, CUDA or None
configure_accel_frontend() # ArrayFire or ViennaCL

if(ENABLE_MPI)
  configure_mpi()
endif()



message(STATUS "Algebra libs: ${ALGEBRA_LIBS}")

if(${ENABLE_PROFILING})
    add_definitions(-DPROFILING)
endif()

add_definitions(-DSMURFF_FLOAT_TYPE=${SMURFF_FLOAT_TYPE})
message(STATUS "Using '${SMURFF_FLOAT_TYPE}' as main computing type")

# support for running "make test" (or alike)

enable_testing()

# unittests
add_test( unittests tests )

if(${EXTERNAL_DATA})
    file(DOWNLOAD  http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm 
            ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-346targets.mm
            EXPECTED_HASH SHA256=10c3e1f989a7a415a585a175ed59eeaa33eff66272d47580374f26342cddaa88
        )

    file(DOWNLOAD http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compound-feat.mm
            ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-compound-feat.mm
            EXPECTED_HASH SHA256=f9fe0d296272ef26872409be6991200dbf4884b0cf6c96af8892abfd2b55e3bc
        )

    file(DOWNLOAD http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compounds.csv
            ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-compounds.csv
            EXPECTED_HASH SHA256=e8f045a67ee149c6100684e07920036de72583366596eb5748a79be6e3b96f7c
        )

    file(DOWNLOAD http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-proteins-uniprot.csv
            ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-proteins-uniprot.csv
            EXPECTED_HASH SHA256=224b1b44abcab8448b023874f4676af30d64fe651754144f9cbdc67853b76ea8
        )

    add_test(chembl_bmpf smurff --num-latent=4 --burnin=2 --nsamples=2
        --train ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-346targets.mm
        )

    add_test(chembl_macau smurff --num-latent=4 --burnin=2 --nsamples=2
        --train ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-346targets.mm
        --row-features ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-compound-feat.mm
        )
endif()

include_directories(cpp)

# CONFIGURE TARGETS

message(STATUS "Creating folder structure...")

SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

# libraries
add_subdirectory (cpp/SmurffCpp)

# main executable
add_subdirectory (cpp/Smurff)

# MPI
if(ENABLE_MPI)
  add_subdirectory (cpp/SmurffMPI)
endif()

# tests
add_subdirectory (cpp/Tests)

# python
if(ENABLE_PYTHON)
    configure_python()
    add_subdirectory (python/smurff)
endif()

