
macro(configure_accel_backend)
  message("Looking for acceleration back-end: ${ACCEL_BACKEND}")
  if(${ACCEL_BACKEND} STREQUAL "CUDA")
    cmake_policy(SET CMP0074 NEW)
    find_package(CUDA REQUIRED)
    set(ALGEBRA_LIBS ${ALGEBRA_LIBS} ${CUDA_LIBRARIES})
  elseif(${ACCEL_BACKEND} STREQUAL "OPENCL")
    find_package(OPENCL REQUIRED)
    set(ALGEBRA_LIBS ${ALGEBRA_LIBS} ${OPENCL_LIBRARIES})
  elseif(NOT ${ACCEL_BACKEND} STREQUAL "NONE")
    message(FATAL_ERROR "ACCEL_BACKEND should not be ${ACCEL_BACKEND}")
  endif()
endmacro(configure_accel_backend)

macro(configure_accel_frontend)
  message("Looking for acceleration front-end: ${ACCEL_FRONTEND}")
  if(${ACCEL_FRONTEND} STREQUAL "ARRAYFIRE")
    find_package(ArrayFire REQUIRED)
    include_directories(${ArrayFire_INCLUDE_DIRS})
    add_definitions(-DUSE_ARRAYFIRE)

    if(${ACCEL_BACKEND} STREQUAL "CUDA")
      set(ALGEBRA_LIBS ${ALGEBRA_LIBS} ArrayFire::afcuda)
    elseif(${ACCEL_BACKEND} STREQUAL "OPENCL")
      set(ALGEBRA_LIBS ${ALGEBRA_LIBS} ArrayFire::afopencl)
    endif()
  elseif(${ACCEL_FRONTEND} STREQUAL "VIENNACL")
    find_package(ViennaCL REQUIRED)
    include_directories(${ViennaCL_INCLUDE_DIRS})
    add_definitions(-DUSE_VIENNACL)
  elseif(NOT ${ACCEL_FRONTEND} STREQUAL "NONE")
    message(FATAL_ERROR "ACCEL_FRONTEND should not be ${ACCEL_FRONTEND}")
  endif()
endmacro(configure_accel_frontend)

macro(report_accel)
  if(${ACCEL_BACKEND} STREQUAL "NONE" AND ${ACCEL_FRONTEND} STREQUAL "NONE")
    message("No acceleration used")
  elseif(${ACCEL_BACKEND} STREQUAL "NONE" OR ${ACCEL_FRONTEND} STREQUAL "NONE")
    message(FATAL_ERROR "Invalid acceleration combination: ${ACCEL_FRONTEND} + ${ACCEL_BACKEND}.
Valid combinations are:
  - NONE + NONE
  - ArrayFire + CUDA
  - ArrayFire + OpenCL
  - ViennaCL + CUDA
  - ViennaCL + OpenCL
")
  else()
    message("Using: ${ACCEL_FRONTEND} + ${ACCEL_BACKEND}")
  endif()
endmacro(report_accel)