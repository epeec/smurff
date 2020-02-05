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

#LINK LIBRARIES
target_link_libraries (mpi_smurff smurff-cpp smurff-utils
                                  ${OpenMP_CXX_LIBRARIES}
                                  ${Boost_LIBRARIES}
                                  ${ALGEBRA_LIBS}
                                  HighFive
                                  ${MPI_LIBRARIES})

set_target_properties(mpi_smurff PROPERTIES COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}")
set_target_properties(mpi_smurff PROPERTIES LINK_FLAGS "${MPI_C_LINK_FLAGS}")

include_directories(${MPI_C_INCLUDE_PATH})

endif()
