#!/usr/bin/env bash

set -e

cd python/smurff


for PYVER in "cp35-cp35m" "cp36-cp36m" "cp37-cp37m" "cp38-cp38" ; do
  PYPREFIX="/opt/python/${PYVER}"
  CMAKE_ARGS="-DBOOST_INCLUDEDIR=/usr/include/boost169 -DBOOST_LIBRARYDIR=/usr/lib64/boost169
     -DPYTHON_INCLUDE_DIR=${PYPREFIX}/include"
  "${PYPREFIX}/bin/pip" install Cython numpy 
  "${PYPREFIX}/bin/python" setup.py --extra-cmake-args "$CMAKE_ARGS" bdist_wheel
done

find dist -name "*.whl" -exec auditwheel repair {} \;
