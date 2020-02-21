FILE (GLOB UTIL_FILES   "TruncNorm.h"
                        "InvNormCdf.h"
                        "counters.h"
                        "linop.h"
                        "omp_util.h"
                        "Error.h"
                        "ThreadVector.hpp"
                        "StringUtils.h"
                        "Tensor.h"

                        "TruncNorm.cpp"
                        "InvNormCdf.cpp"
                        "counters.cpp"
                        "omp_util.cpp"
                        "StringUtils.cpp"
                        "Tensor.cpp"
                        )

source_group ("Utils" FILES ${UTIL_FILES})

#SETUP OUTPUT
add_library (smurff-utils STATIC ${UTIL_FILES})