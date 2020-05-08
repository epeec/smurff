#!/bin/sh

sed -i -e 's/MKL::Shared/MKL::RT/g' src/backend/*/CMakeLists.txt src/api/unified/CMakeLists.txt

mkdir build && cd build
export MLKROOT=$PREFIX
cmake \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
    -DUSE_CPU_MKL=ON \
    -DAF_BUILD_CPU=ON \
    -DAF_BUILD_CUDA=OFF \
    -DAF_BUILD_OPENCL=OFF \
    -DAF_BUILD_UNIFIED=OFF \
    -DAF_BUILD_EXAMPLES=OFF \
    -DBUILD_TESTING=OFF \
    ..
make -j4
make install
rm -r $PREFIX/share/ArrayFire/examples
rm -r $PREFIX/LICENSES
