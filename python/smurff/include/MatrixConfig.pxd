from libc.stdint cimport *
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp cimport bool

from TensorConfig cimport TensorConfig
from NoiseConfig cimport NoiseConfig

cdef extern from "<SmurffCpp/Configs/MatrixConfig.h>" namespace "smurff":
    cdef cppclass MatrixConfig(TensorConfig):
        #
        # Dense double matrix constructor
        #
        MatrixConfig(uint64_t nrow, uint64_t ncol,
                     const double* values,
                     const NoiseConfig& noiseConfig) except +

        #
        # Sparse double matrix constructors
        #
        MatrixConfig(uint64_t nrow, uint64_t ncol, uint64_t nnz,
                     const uint32_t* rows,
                     const uint32_t* cols, 
                     const double* values,
                     const NoiseConfig& noiseConfig, bool isScarse) except +
