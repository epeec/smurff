#!/bin/sh
#
# Start this script in a Docker container like this:
#
#  docker run -v $(git rev-parse --show-toplevel):/smurff -ti smurff1604 /smurff/ci/ubuntu1604/build_script.sh
#
# where smurff1604 is the image name


set -e
set -x

rm -rf /build  && mkdir /build && cd /build
cmake /smurff  -DENABLE_PYTHON=OFF 
make -j${CPU_COUNT}
./bin/tests
