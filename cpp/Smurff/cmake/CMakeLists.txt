#SETUP PROJECT
FILE (GLOB SOURCE_FILES "../smurff.cpp")

source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_executable (smurff ${HEADER_FILES} ${SOURCE_FILES})
install(TARGETS smurff RUNTIME DESTINATION bin)

#LINK LIBRARIES
target_link_libraries (smurff smurff-cpp smurff-utils
                                  ${OpenMP_CXX_LIBRARIES}
                                  ${Boost_LIBRARIES}
                                  ${BOOST_RANDOM_LIBRARIES}
                                  ${ALGEBRA_LIBS}
                                  ${CMAKE_THREAD_LIBS_INIT})

#SETUP INCLUDES
include_directories(../)
include_directories(../..)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${BOOST_RANDOM_INCLUDE_DIRS})
