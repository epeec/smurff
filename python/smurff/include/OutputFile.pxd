from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from SaveState cimport SaveState

cdef extern from "<SmurffCpp/Utils/OutputFile.h>" namespace "smurff":
    cdef cppclass OutputFile:
        string getFullPath() 
        string getOptionsFileName() 
        vector[shared_ptr[SaveState]] openSampleSteps()
