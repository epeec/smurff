#!/bin/sh
#
# Start this script in a Docker container like this:
#
#  docker run -v $(git rev-parse --show-toplevel):/smurff -ti smurff1804 /smurff/ci/ubuntu1804/build_script.sh
#
# where smurff1804 is the image name


set -e
set -x

cd /smurff/build
rm -rf docker1804 && mkdir docker1804 && cd docker1804
cmake ../..  -DHIGHFIVE_USE_BOOST=OFF
make -j2
./bin/tests '~[random]'
