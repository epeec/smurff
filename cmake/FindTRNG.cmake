# Defines the following variables:
#   - TRNG_FOUND
#   - TRNG_LIBRARY
#   - TRNG_INCLUDE_DIRS

# Find the header files
find_path(TRNG_INCLUDE_DIRS trng/ 
  HINTS ${TRNG_DIR} ${CMAKE_BINARY_DIR}/external/trng4 $ENV{TRNG_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with TRNG header.")
find_path(TRNG_INCLUDE_DIRS trng/)

# Find the library
find_library(TRNG_LIBRARY trng4 
  HINTS ${TRNG_DIR} ${CMAKE_BINARY_DIR}/external/trng4 $ENV{TRNG_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The TRNG library.")
find_library(TRNG_LIBRARY trng4)