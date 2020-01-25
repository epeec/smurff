from libc.stdint cimport *
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp cimport bool

from TensorConfig cimport TensorConfig
from NoiseConfig cimport NoiseConfig

cdef extern from "<SmurffCpp/Configs/MatrixConfig.h>" namespace "smurff":
    cdef cppclass MatrixConfig(TensorConfig):
        #
        # Dense double matrix constructos
        #
        MatrixConfig(uint64_t nrow, uint64_t ncol,
                     const vector[double]& values,
                     const NoiseConfig& noiseConfig) except +

        #
        # Sparse double matrix constructors
        #
        MatrixConfig(uint64_t nrow, uint64_t ncol,
                     const vector[uint32_t]& rows,
                     const vector[uint32_t]& cols, 
                     const vector[double]& values,
                     const NoiseConfig& noiseConfig, bool isScarse) except +
