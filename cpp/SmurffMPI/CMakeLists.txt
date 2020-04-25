if(${MPI_C_FOUND})

FILE (GLOB HEADER_FILES "MPISession.h"
                        "MPIMacauPrior.h"
                        "MPIMacauPrior.cpp"
                        "MPIPriorFactory.h"
                       )
source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "mpi_smurff.cpp"
                        "MPISession.cpp"
                        "MPIPriorFactory.cpp"
                        )
source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_executable (mpi_smurff ${HEADER_FILES} ${SOURCE_FILES})
install(TARGETS mpi_smurff RUNTIME DESTINATION bin)

if (${OPENMP_FOUND})
    target_link_libraries (mpi_smurff smurff-cpp-mt ${OpenMP_CXX_LIBRARIES})
    set_target_properties(mpi_smurff PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
else()
    target_link_libraries (mpi_smurff smurff-cpp-st)
endif()

target_link_libraries (mpi_smurff ${Boost_LIBRARIES}
                                  ${ALGEBRA_LIBS}
                                  HighFive
                                  ${MPI_LIBRARIES})

set_target_properties(mpi_smurff PROPERTIES COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}")
set_target_properties(mpi_smurff PROPERTIES LINK_FLAGS "${MPI_C_LINK_FLAGS}")

include_directories(${MPI_C_INCLUDE_PATH})

endif()
