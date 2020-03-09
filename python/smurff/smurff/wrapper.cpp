#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>


#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Configs/NoiseConfig.h>
#include <SmurffCpp/Configs/DataConfig.h>

#include <SmurffCpp/Sessions/PythonSession.h>

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

    py::class_<smurff::NoiseConfig>(m, "NoiseConfig")
        .def(py::init<>())
        .def("setNoiseType", py::overload_cast<std::string>(&smurff::NoiseConfig::setNoiseType))
        .def("setPrecision", &smurff::NoiseConfig::setPrecision)
        .def("setSnInit", &smurff::NoiseConfig::setSnInit)
        .def("setSnMax", &smurff::NoiseConfig::setSnMax)
        .def("setThreshold", &smurff::NoiseConfig::setThreshold)
        ;

    py::class_<smurff::StatusItem>(m, "StatusItem")
        .def(py::init<>())
        .def("__str__", &smurff::StatusItem::asString)
        .def_readonly("phase", &smurff::StatusItem::phase)
        .def_readonly("iter", &smurff::StatusItem::iter)
        .def_readonly("rmse_avg", &smurff::StatusItem::rmse_avg)
        .def_readonly("rmse_1sample", &smurff::StatusItem::rmse_1sample)
        .def_readonly("train_rmse", &smurff::StatusItem::train_rmse)
        .def_readonly("auc_avg", &smurff::StatusItem::auc_avg)
        .def_readonly("auc_1sample", &smurff::StatusItem::auc_1sample)
        .def_readonly("elapsed_iter", &smurff::StatusItem::elapsed_iter)
        .def_readonly("elapsed_total", &smurff::StatusItem::elapsed_total)
        .def_readonly("elapsed_total", &smurff::StatusItem::elapsed_total)
        .def_readonly("nnz_per_sec", &smurff::StatusItem::nnz_per_sec)
        .def_readonly("samples_per_sec", &smurff::StatusItem::samples_per_sec)
        ;

    py::class_<smurff::DataConfig>(m, "DataConfig")
        .def(py::init<>())
        .def("setData", py::overload_cast<const smurff::Matrix &>(&smurff::DataConfig::setData))
        .def("setData", py::overload_cast<const smurff::SparseMatrix &, bool>(&smurff::DataConfig::setData))
        .def("setData", py::overload_cast<const smurff::DenseTensor &>(&smurff::DataConfig::setData))
        .def("setData", py::overload_cast<const smurff::SparseTensor &, bool>(&smurff::DataConfig::setData))
        .def("setNoiseConfig", &smurff::DataConfig::setNoiseConfig)
        ;

    py::class_<smurff::PythonSession>(m, "PythonSession")
        .def(py::init<>())
        .def("__str__", &smurff::ISession::infoAsString)
        .def("getStatus", &smurff::TrainSession::getStatus)
        .def("fromConfig", &smurff::TrainSession::fromConfig)
        .def("init", &smurff::TrainSession::init)
        .def("step", &smurff::PythonSession::step)
        .def("interrupted", &smurff::PythonSession::interrupted)
        ;
}
