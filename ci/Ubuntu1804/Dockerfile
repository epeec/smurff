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
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/tvandera/HighFive.git && \
    cd HighFive && mkdir build && cd build && \
    cmake .. && \
    make -j2 && make install && \
    cd ../../ && rm -r HighFive


