# set (CYTHON_FLAGS "--directive" "embedsignature=True")
# add_cython_target(wrapper CXX)
# add_library(wrapper MODULE ${wrapper})
# python_extension_module(wrapper)

pybind11_add_module(wrapper wrapper.cpp)
# add_library(wrapper MODULE wrapper.cpp)

#LINK LIBRARIES
target_link_libraries (wrapper PRIVATE smurff-cpp smurff-utils pybind11::module
                            ${ALGEBRA_LIBS}
                            ${OpenMP_CXX_LIBRARIES})

