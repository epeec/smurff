
macro(configure_lapack)
  message ("Dependency check for lapack...")
  find_package(LAPACK REQUIRED)
  find_package(LAPACKE REQUIRED)
  add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)
  message(STATUS LAPACK: ${LAPACK_LIBRARIES})
endmacro(configure_lapack)

macro(configure_openblas)
  message ("Dependency check for openblas...")
  set(BLA_VENDOR "OpenBLAS")
  find_package( BLAS REQUIRED )

  add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)

  message(STATUS BLAS: ${BLAS_LIBRARIES} )
endmacro(configure_openblas)

macro(configure_mkl)
  message ("Dependency check for MKL (using MKL SDL)...")
  find_library (MKL_LIBRARIES "mkl_rt" HINTS ENV LD_LIBRARY_PATH REQUIRED)
  find_path (MKL_INCLUDE_DIR "mkl.h" HINTS ENV CPATH REQUIRED)

  if (NOT MKL_INCLUDE_DIR OR NOT MKL_LIBRARIES)
    message(FATAL_ERROR "MKL not found (mkl.h or mkl_rt lib)")
  endif()

  include_directories(${MKL_INCLUDE_DIR})

  # make sure we link with iomp5 and not gomp
  list(FIND OpenMP_CXX_LIB_NAMES "gomp" GNU_OPENMP)
  list(FIND OpenMP_CXX_LIB_NAMES "omp" LLVM_OPENMP)
  if(NOT GNU_OPENMP EQUAL -1)
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-openmp")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fno-openmp")
      find_library (IOMP_LIBRARIES "iomp5" HINTS ENV LD_LIBRARY_PATH REQUIRED)
      set(MKL_LIBRARIES "${MKL_LIBRARIES};${IOMP_LIBRARIES}")
  elseif(NOT LLVM_OPENMP EQUAL -1)
      message(ERROR "Please use iomp when using clang/llvm compiler, not omp")
  endif()
  
  add_definitions(-DEIGEN_USE_MKL_ALL)
  
  message(STATUS "MKL_LIBRARIES ${MKL_LIBRARIES}" )
  message(STATUS "MKL_INCLUDE_DIR ${MKL_INCLUDE_DIR}" )
endmacro(configure_mkl)

macro(configure_algebra)
  if(${ALGEBRA_LIB_NAME} STREQUAL "MKL")
    configure_mkl()
    set(ALGEBRA_LIBS ${ALGEBRA_LIBS} ${MKL_LIBRARIES})
  elseif(${ALGEBRA_LIB_NAME} STREQUAL "OPENBLAS")
    configure_openblas()
    set(ALGEBRA_LIBS ${ALGEBRA_LIBS} ${BLAS_LIBRARIES})
  elseif(${ALGEBRA_LIB_NAME} STREQUAL "LAPACKE")
    configure_lapack()
    set(ALGEBRA_LIBS ${ALGEBRA_LIBS} ${LAPACKE_LIBRARIES} ${LAPACK_LIBRARIES})
  else()
    message(FATAL_ERROR "Unknown ALGEBRA_LIB_NAME: ${ALGEBRA_LIB_NAME}")
  endif()
endmacro(configure_algebra)