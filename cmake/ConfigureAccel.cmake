
macro(configure_accel_backend)
  message("Looking for acceleration back-end: ${ACCEL_BACKEND}")
  if(${ACCEL_BACKEND} STREQUAL "CUDA")
    cmake_policy(SET CMP0074 NEW)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(ALGEBRA_LIBS ${ALGEBRA_LIBS} ${CUDA_LIBRARIES})
  elseif(${ACCEL_BACKEND} STREQUAL "OpenCL")
    find_package(OpenCL REQUIRED)
    if (APPLE)
      add_definitions(-DCL_SILENCE_DEPRECATION)
    endif()
    include_directories(${OpenCL_INCLUDE_DIRS})
    set(ALGEBRA_LIBS ${ALGEBRA_LIBS} ${OPENCL_LIBRARIES})
  elseif(NOT ${ACCEL_BACKEND} STREQUAL "None")
    message(FATAL_ERROR "ACCEL_BACKEND should not be ${ACCEL_BACKEND}
    Valid options: CUDA, OpenCL, or None
    ")
  endif()
endmacro(configure_accel_backend)

macro(configure_accel_frontend)
  message("Looking for acceleration front-end: ${ACCEL_FRONTEND}")
  if(${ACCEL_FRONTEND} STREQUAL "ArrayFire")
    find_package(ArrayFire REQUIRED)
    include_directories(${ArrayFire_INCLUDE_DIRS})
    add_definitions(-DUSE_ARRAYFIRE)

    if(${ACCEL_BACKEND} STREQUAL "CUDA")
      set(ALGEBRA_LIBS ${ALGEBRA_LIBS} ArrayFire::afcuda)
    elseif(${ACCEL_BACKEND} STREQUAL "OPENCL")
      set(ALGEBRA_LIBS ${ALGEBRA_LIBS} ArrayFire::afopencl)
    endif()

  elseif(${ACCEL_FRONTEND} STREQUAL "ViennaCL")
    find_package(ViennaCL REQUIRED)
    include_directories(${ViennaCL_INCLUDE_DIRS})
    add_definitions(-DUSE_VIENNACL)

    if(${ACCEL_BACKEND} STREQUAL "CUDA")
      add_definitions(-DVIENNACL_WITH_CUDA)
    elseif(${ACCEL_BACKEND} STREQUAL "OpenCL")
      add_definitions(-DVIENNACL_WITH_OPENCL)
    endif()

  elseif(NOT ${ACCEL_FRONTEND} STREQUAL "None")
    message(FATAL_ERROR "ACCEL_FRONTEND should not be ${ACCEL_FRONTEND}
    Valid options are: ArrayFire, ViennaCL or None
    ")
  endif()
endmacro(configure_accel_frontend)

macro(report_accel)
  if(${ACCEL_BACKEND} STREQUAL "NONE" AND ${ACCEL_FRONTEND} STREQUAL "NONE")
    message("No acceleration used")
  elseif(${ACCEL_BACKEND} STREQUAL "NONE" OR ${ACCEL_FRONTEND} STREQUAL "NONE")
    message(FATAL_ERROR "Invalid acceleration combination: ${ACCEL_FRONTEND} + ${ACCEL_BACKEND}.
Valid combinations are (Frontend + Backend):
  - None + None
  - ArrayFire + CUDA
  - ArrayFire + OpenCL
  - ViennaCL + CUDA
  - ViennaCL + OpenCL
")
  else()
    message("Using: ${ACCEL_FRONTEND} + ${ACCEL_BACKEND}")
  endif()
endmacro(report_accel)