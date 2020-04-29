FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       cmake \
       git g++ python3 \
       libblas-dev \
       liblapack-dev \
       liblapacke-dev \
       libopenmpi-dev \
       openmpi-bin \
       libeigen3-dev \
       libboost-all-dev \
       ca-certificates \
       libhdf5-dev \
       gdb vim wget\
       libhdf5-dev \
       python3-numpy python3-pybind11 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O HighFive.tar.gz https://github.com/BlueBrain/HighFive/archive/v2.2.tar.gz && \
    tar xzf HighFive.tar.gz && \
    cd HighFive* && mkdir build && cd build && \
    cmake .. && \
    make -j2 && make install && \
    cd ../../ && rm -r HighFive*

