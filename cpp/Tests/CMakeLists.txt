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

if (${OPEMP_FOUND})
add_executable (tests-mt ${HEADER_FILES} ${SOURCE_FILES})
install(TARGETS tests-mt RUNTIME DESTINATION libexec)

target_link_libraries (tests-mt smurff-cpp-mt
                                  ${OpenMP_CXX_LIBRARIES}
                                  ${Boost_LIBRARIES}
                                  ${ALGEBRA_LIBS}
                                  HighFive
                      )
set_target_properties(tests-mt PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
endif()

add_executable (tests-st ${HEADER_FILES} ${SOURCE_FILES})
install(TARGETS tests-st RUNTIME DESTINATION libexec)

target_link_libraries (tests-st smurff-cpp-st
                                  ${Boost_LIBRARIES}
                                  ${ALGEBRA_LIBS}
                                  HighFive
                      )
 
