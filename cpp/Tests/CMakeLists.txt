FILE (GLOB HEADER_FILES "catch.hpp" "Tests.h")
source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "tests.cpp"
                        "TestsMatrixIO.cpp"
                        "TestsTensorIO.cpp"
                        "TestsMatrixConfig.cpp"
                        "TestsTensorConfig.cpp"
                        "TestsPVec.cpp"
                        "TestsTensorData.cpp"
                        "TestsMatrixUtils.cpp"
                        "TestsLinop.cpp"
                        "TestsInputData.cpp"
                        "TestsSmurff.cpp"
                        "TestsMatrixVsTensor.cpp"
                        "TestsPredict.cpp"
                        "TestsSparseSideInfo.cpp"
                        )
                        
source_group ("Source Files" FILES ${SOURCE_FILES})

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

if("${SMURFF_FLOAT_TYPE}" STREQUAL "float")
    include_directories("expected_float_results")
elseif("${SMURFF_FLOAT_TYPE}" STREQUAL "double")
    include_directories("expected_double_results")
else()
    message(FATAL_ERROR "Unknown float type (${SMURFF_FLOAT_TYPE}) for tests - only 'double' and 'float' supported")
endif()
