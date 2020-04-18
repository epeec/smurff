#!/bin/sh

set -e

rm -f *.h5

# export DYLD_INSERT_LIBRARIES=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/11.0.3/lib/darwin/libclang_rt.asan_osx_dynamic.dylib

python ../hardcoded/run_ini.py --verbose 1 \
    --seed 1234 --num-threads 1 \
    --burnin 2 --nsamples 2 \
    --train chembl-IC50-346targets-100compounds-train.sdm \
    --test chembl-IC50-346targets-100compounds-test.sdm \
    --row-features chembl-IC50-100compounds-feat-dense.ddm \
    --prior macau normal --save-name macau-py.h5 run >py.log

python ../hardcoded/run_ini.py --verbose 1 \
    --seed 1234 --num-threads 1 \
    --burnin 2 --nsamples 2 \
    --train chembl-IC50-346targets-100compounds-train.sdm \
    --test chembl-IC50-346targets-100compounds-test.sdm \
    --row-features chembl-IC50-100compounds-feat-dense.ddm \
    --prior macau normal --save-name macau.h5 save

smurff --restore-from macau.h5 --verbose 1 --save-name macau-out.h5 >cmd.log


