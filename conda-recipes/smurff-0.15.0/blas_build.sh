#!/bin/bash
rm -rf build 
mkdir build
cd build

if [ -n "$OSX_ARCH" ]
then
    OPENMP_FLAGS="-DOpenMP_CXX_FLAG=-fopenmp=libiomp5 -DOpenMP_C_FLAG=-fopenmp=libiomp5"
else
    OPENMP_FLAGS=""
fi


if [ "$1" == "mkl" ]
then
    BLAS_FLAGS="-DENABLE_MKL=ON"
elif [ $1 == "openblas" ]
then
    BLAS_FLAGS="-DENABLE_OPENBLAS=ON"
else
    exit -1
fi

cmake ../lib/smurff-cpp/cmake -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_MPI=OFF \
    -DCMAKE_INSTALL_PREFIX=$PREFIX $BLAS_FLAGS $OPENMP_FLAGS

make -j$CPU_COUNT
make install
cd python/Smurff

$PYTHON setup.py install --single-version-externally-managed --record=record.txt
