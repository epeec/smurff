set (SCRIPT_DIR "${CMAKE_SOURCE_DIR}/cmake/")
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

macro(configure_pthreads)
    message ("Dependency check for pthreads multi-threading support...")

    if(UNIX)
        find_package(Threads REQUIRED)
        if(Threads_FOUND)
            message(STATUS "Found threading library")
            if(CMAKE_USE_PTHREADS_INIT)
                message(STATUS "Found pthreads " ${CMAKE_THREAD_LIBS_INIT})
            else()
                message(STATUS "Pthreads not found")
            endif()
        else()
            message(STATUS "Threading library not found")
        endif()
    else()
       message(STATUS "Not required on windows")
    endif()
endmacro(configure_pthreads)

macro(configure_mpi)
  message ("Dependency check for mpi...")

  find_package(MPI)
  if(${MPI_C_FOUND})
    message(STATUS "MPI found")
  else()
    message(STATUS "MPI not found")
  endif()
   
endmacro(configure_mpi)

macro(configure_openmp)
  message ("Dependency check for OpenMP")

  find_package(OpenMP)
  if(${OPENMP_FOUND})
      message(STATUS "OpenMP found")
      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
      set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")

      include_directories(${OpenMP_CXX_INCLUDE_DIRS})
      include_directories(${OpenMP_C_INCLUDE_DIRS})

      message(STATUS "OpenMP_CXX_LIB_NAMES ${OpenMP_CXX_LIB_NAMES}")
      message(STATUS "OpenMP_CXX_LIBRARY ${OpenMP_CXX_LIBRARY}")
      message(STATUS "OpenMP_CXX_LIBRARIES ${OpenMP_CXX_LIBRARIES}")
      message(STATUS "OpenMP_CXX_INCLUDE_DIRS ${OpenMP_CXX_INCLUDE_DIRS}")
      message(STATUS "OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS}")
  else()
      message(STATUS "OpenMP not found")
  endif()
   
endmacro(configure_openmp)

macro(configure_lapack)
  message ("Dependency check for lapack...")
  find_package(LAPACK REQUIRED)
  find_package(LAPACKE REQUIRED)
  add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)
  message(STATUS LAPACK: ${LAPACK_LIBRARIES})
endmacro(configure_lapack)

macro(configure_openblas)
  message ("Dependency check for openblas...")
  
  if(MSVC)
  set(BLAS_LIBRARIES  $ENV{BLAS_LIBRARIES})
  set(BLAS_INCLUDES $ENV{BLAS_INCLUDES})
  set(BLAS_FOUND ON)
  else()
  set(BLA_VENDOR "OpenBLAS")
  find_package( BLAS REQUIRED )
  endif()

  add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKEyy)

  message(STATUS BLAS: ${BLAS_LIBRARIES} )
 
endmacro(configure_openblas)

macro(configure_mkl)
  message ("Dependency check for MKL (using MKL SDL)...")
  find_library (MKL_LIBRARIES "mkl_rt" HINTS ENV LD_LIBRARY_PATH REQUIRED)
  find_PATH (MKL_INCLUDE_DIR "mkl.h" HINTS ENV CPATH REQUIRED)

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
  
  message(STATUS MKL: ${MKL_LIBRARIES} )
endmacro(configure_mkl)

macro(configure_eigen)
  message ("Dependency check for eigen...")
  
  if(DEFINED ENV{EIGEN3_INCLUDE_DIR})
  SET(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
  else()
  find_package(Eigen3 REQUIRED)
  endif()

  include_directories(${EIGEN3_INCLUDE_DIR})
  
  message(STATUS EIGEN3: ${EIGEN3_INCLUDE_DIR})
endmacro(configure_eigen)

macro(configure_boost)
  message ("Dependency check for boost...")
  if(${ENABLE_BOOST})
      set (Boost_USE_STATIC_LIBS OFF)
      set (Boost_USE_MULTITHREADED ON)

      # find boost random library - optional
      if(ENABLE_BOOST_RANDOM)
        set (BOOST_COMPONENTS random system program_options)
        FIND_PACKAGE(Boost 1.59 EXACT COMPONENTS ${BOOST_COMPONENTS} REQUIRED)
        message("Found Boost random library")
        add_definitions(-DUSE_BOOST_RANDOM)
      else()
        set (BOOST_COMPONENTS system program_options)
        FIND_PACKAGE(Boost COMPONENTS ${BOOST_COMPONENTS} REQUIRED)
      endif()

      message("-- Found Boost_VERSION: ${Boost_VERSION}")
      message("-- Found Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
      message("-- Found Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")
      add_definitions(-DHAVE_BOOST)

      include_directories(${Boost_INCLUDE_DIRS})
  else()
      message("-- Boost library is not enabled")
  endif()
endmacro(configure_boost)

macro(configure_python)
    if(ENABLE_PYTHON)
        find_package(PythonInterp REQUIRED)
        find_package(NumPy REQUIRED)
        find_package(PythonLibs REQUIRED)
        find_package(PythonExtensions REQUIRED)
        find_package(Cython REQUIRED)
    endif()
endmacro(configure_python)
