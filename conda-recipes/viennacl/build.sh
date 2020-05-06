#!/bin/sh
mkdir build && cd build
export OPENCLROOT=/usr/local/cuda
export CXXFLAGS=-Wno-ignored-attributes
cmake .. \
    -DCMAKE_LIBRARY_PATH=${OPENCLROOT}/lib64 \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_BUILD_TYPE=Release 
make -j
make install
