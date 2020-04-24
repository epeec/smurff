FILE (GLOB HEADER_FILES "catch.hpp" "Tests.h")
source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "tests.cpp"
                        "TestsDistribution.cpp"
                        "TestsPVec.cpp"
                        "TestsTensorData.cpp"
                        "TestsLinop.cpp"
                        "TestsInputData.cpp"
                        "TestsSmurff.cpp"
                        "TestsMatrixVsTensor.cpp"
                        "TestsPredict.cpp"
                        "TestsSparseSideInfo.cpp"
                        "TestsSmurff.cpp"
                        "TestsRandom.cpp"
                        )

include_directories("expected_${SMURFF_FLOAT_TYPE}")
                        
source_group ("Source Files" FILES ${SOURCE_FILES} )

#SETUP OUTPUT
add_executable (tests ${HEADER_FILES} ${SOURCE_FILES})
install(TARGETS tests RUNTIME DESTINATION libexec)

#LINK LIBRARIES
target_link_libraries (tests smurff-cpp smurff-utils
                                  ${OpenMP_CXX_LIBRARIES}
                                  ${Boost_LIBRARIES}
                                  ${ALGEBRA_LIBS}
                                  HighFive
                      )
