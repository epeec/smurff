#!/bin/sh
#
# Start this script in a Docker container like this:
#
#  docker run -v $(git rev-parse --show-toplevel):/smurff -ti smurff1910 /smurff/ci/ubuntu1910/build_script.sh
#
# where smurff1910 is the image name


set -e
set -x

cd /smurff/build
rm -rf docker1910 && mkdir docker1910 && cd docker1910
cmake ../.. 
make -j2
./bin/tests '~[random]'
