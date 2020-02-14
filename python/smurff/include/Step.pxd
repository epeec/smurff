from libc.stdint cimport *
from libcpp.string cimport string

cdef extern from "<SmurffCpp/Utils/Step.h>" namespace "smurff":
    cdef cppclass Step:
        string getStepIniFileName()
        string getModelFileName(uint64_t index)
        string getMuFileName(uint64_t index)
        string getLinkMatrixFileName(uint32_t mode)
        string getPredFileName()
        string getPredStateFileName()
        int32_t getIsample()
