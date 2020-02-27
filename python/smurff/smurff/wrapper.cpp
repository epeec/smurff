#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "SmurffCpp/Configs/Config.h"

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

// wrap as Python module
PYBIND11_MODULE(wrapper, m)
{
    m.doc() = "SMURFF Python Interface";

    py::class_<smurff::Config>(m, "Config")
        .def(py::init<>())
        .def("setTrain", &smurff::Config::setTrain)
        .def("setTest", &smurff::Config::setTest)
        .def("addSideInfoConfig", &smurff::Config::addSideInfoConfig)
        .def("setPriorTypes", py::overload_cast<std::vector<std::string>>(&smurff::Config::setPriorTypes))
        .def("setModelInitType", py::overload_cast<std::string>(&smurff::Config::setModelInitType))
        .def("setSavePrefix", &smurff::Config::setSavePrefix)
        .def("setSaveExtension", &smurff::Config::setSaveExtension)
        .def("setSaveFreq", &smurff::Config::setSaveFreq)
        .def("setRandomSeed", &smurff::Config::setRandomSeed)
        .def("setVerbose", &smurff::Config::setVerbose)
        .def("setBurnin", &smurff::Config::setBurnin)
        .def("setNSamples", &smurff::Config::setNSamples)
        .def("setNumLatent", &smurff::Config::setNumLatent)
        .def("setNumThreads", &smurff::Config::setNumThreads)
        .def("setThreshold", &smurff::Config::setThreshold)
        ;
}
