FILE (GLOB HEADER_FILES "catch.hpp")
source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "tests.cpp"
                        "TestsMatrixIO.cpp"
                        "TestsTensorIO.cpp"
                        "TestsMatrixConfig.cpp"
                        "TestsTensorConfig.cpp"
     	  	   	        "TestsPVec.cpp"
                        "TestsTensor.cpp"
                        "TestsMatrixUtils.cpp"
                        "TestsLinop.cpp"
                        "TestsSmurff.cpp"
                        "TestsSparseSideInfo.cpp"
                        )
                        
source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_executable (tests ${HEADER_FILES} ${SOURCE_FILES})
SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
install(TARGETS tests RUNTIME DESTINATION libexec)

#LINK LIBRARIES
target_link_libraries (tests smurff-cpp smurff-utils
                                  ${OpenMP_CXX_LIBRARIES}
                                  ${ALGEBRA_LIBS}
                                  ${CMAKE_THREAD_LIBS_INIT})

if("${SMURFF_FLOAT_TYPE}" STREQUAL "float")
    include_directories("expected_float_results")
elseif("${SMURFF_FLOAT_TYPE}" STREQUAL "double")
    include_directories("expected_double_results")
else()
    message(FATAL_ERROR "Unknown float type (${SMURFF_FLOAT_TYPE}) for tests - only 'double' and 'float' supported")
endif()