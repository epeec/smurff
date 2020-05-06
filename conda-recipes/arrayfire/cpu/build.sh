#!/bin/sh
mkdir build && cd build
export MLKROOT=$PREFIX
cmake \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CPU_MKL=ON \
    -DAF_BUILD_CPU=ON \
    -DAF_BUILD_CUDA=OFF \
    -DAF_BUILD_OPENCL=OFF \
    -DAF_BUILD_EXAMPLES=OFF \
    -DBUILD_TESTING=OFF \
    ..
make -j
make install
rm -r $PREFIX/share/ArrayFire/examples
rm -r $PREFIX/LICENSES
