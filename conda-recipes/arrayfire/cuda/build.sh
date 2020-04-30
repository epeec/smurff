#!/bin/sh
mkdir build && cd build

export CXXFLAGS=$(echo $CXXFLAGS | sed -e s/std=c++17/std=c++14/g)

ln -s /usr/include/cublas_v2.h $PREFIX/include/
ln -s /usr/include/GL $PREFIX/include/

cmake \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
    -DUSE_CPU_MKL=OFF \
    -DAF_BUILD_CPU=OFF \
    -DAF_BUILD_CUDA=ON \
    -DAF_BUILD_OPENCL=OFF \
    -DAF_BUILD_UNIFIED=OFF \
    -DAF_BUILD_EXAMPLES=OFF \
    -DBUILD_TESTING=OFF \
    ..
make -j
make install

rm -r $PREFIX/share/ArrayFire/examples
rm -r $PREFIX/LICENSES
rm    $PREFIX/include/cublas_v2.h
rm    $PREFIX/include/GL
