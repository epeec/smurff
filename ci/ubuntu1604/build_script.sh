#!/bin/sh
#
# Start this script in a Docker container like this:
#
#  docker run -v $(git rev-parse --show-toplevel):/smurff -ti smurff1604 /smurff/ci/ubuntu1604/build_script.sh
#
# where smurff1604 is the image name


set -e

cd /smurff/build
rm -rf docker1604
mkdir docker1604
cd docker1604
cmake ../..  -DHIGHFIVE_USE_BOOST=OFF -DENABLE_PYTHON=OFF -DBOOST_RANDOM_VERSION=1.58
make -j2
./bin/tests
