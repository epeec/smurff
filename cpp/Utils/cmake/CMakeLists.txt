FILE (GLOB UTIL_FILES   "../TruncNorm.h"
                        "../InvNormCdf.h"
                        "../counters.h"
                        "../linop.h"
                        "../omp_util.h"
                        "../Error.h"
                        "../ThreadVector.hpp"
                        "../StringUtils.h"

                        "../TruncNorm.cpp"
                        "../InvNormCdf.cpp"
                        "../counters.cpp"
                        "../omp_util.cpp"
                        "../StringUtils.cpp"
                        )

source_group ("Utils" FILES ${UTIL_FILES})

#SETUP OUTPUT
add_library (smurff-utils STATIC ${UTIL_FILES})

#SETUP INCLUDES
include_directories(../..)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${BOOST_RANDOM_INCLUDE_DIRS})


