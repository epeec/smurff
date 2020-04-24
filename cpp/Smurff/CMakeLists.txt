FILE (GLOB SOURCE_FILES "smurff.cpp")

source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_executable (smurff ${HEADER_FILES} ${SOURCE_FILES})
install(TARGETS smurff RUNTIME DESTINATION bin)

#LINK LIBRARIES
target_link_libraries (smurff smurff-cpp smurff-utils
                                  ${OpenMP_CXX_LIBRARIES}
                                  ${Boost_LIBRARIES}
                                  ${ALGEBRA_LIBS}
                                  HighFive
                      )