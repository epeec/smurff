FILE (GLOB SOURCE_FILES "smurff.cpp")

source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_executable (smurff ${HEADER_FILES} ${SOURCE_FILES})
install(TARGETS smurff RUNTIME DESTINATION bin)

if (${OPENMP_FOUND})
    target_link_libraries (smurff smurff-cpp-mt ${OpenMP_CXX_LIBRARIES})
    set_target_properties(smurff PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
else()
    target_link_libraries (smurff smurff-cpp-st)
endif()

target_link_libraries (smurff ${Boost_LIBRARIES} ${ALGEBRA_LIBS} HighFive)
