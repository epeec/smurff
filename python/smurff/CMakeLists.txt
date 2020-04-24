pybind11_add_module(wrapper src/smurff/wrapper.cpp)

#LINK LIBRARIES
target_link_libraries (wrapper PRIVATE smurff-cpp smurff-utils pybind11::module
                            ${ALGEBRA_LIBS}
                            ${TRNG_LIBRARY}
                            HighFive
                            ${OpenMP_CXX_LIBRARIES}

                     )
