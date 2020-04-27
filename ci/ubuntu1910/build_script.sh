#!/bin/sh
#
# Start this script in a Docker container like this:
#
#  docker run -eCPU_COUNT=2 -v $(git rev-parse --show-toplevel):/smurff -ti smurff1910 /smurff/ci/ubuntu1910/build_script.sh
#
# where smurff1910 is the image name


set -e
set -x

rm -rf /build  && mkdir /build && cd /build
git clone /smurff 
cd smurff/python/smurff
python3 setup.py install --install-binaries --extra-build-args "-j ${CPU_COUNT}"
/usr/local/libexec/tests
python3 -m unittest discover test
