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
        // add data
        .def("setTrain", &smurff::Config::setTrain)
        .def("setTest", &smurff::Config::setTest)
        .def("addSideInfoConfig", &smurff::Config::addSideInfoConfig)

        // set scalar config values
        .def("setPriorTypes", py::overload_cast<std::vector<std::string>>(&smurff::Config::setPriorTypes))
        .def("setModelInitType", py::overload_cast<std::string>(&smurff::Config::setModelInitType))
        .def("setOutputFilename", &smurff::Config::setOutputFilename)
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
        .def(py::init<const std::string, double, double, double, double>(),
           py::arg("noise_type") = "fixed",
           py::arg("precision") = 5.0,
           py::arg("sn_init") = 1.0,
           py::arg("sn_max") = 10.0,
           py::arg("threshold") = 0.5
        )  
        ;

    py::class_<smurff::StatusItem>(m, "StatusItem", "Short set of parameters indicative for the training progress.")
        .def(py::init<>())
        .def("__str__", &smurff::StatusItem::asString)
        .def_readonly("phase", &smurff::StatusItem::phase, "{ \"Burnin\", \"Sampling\" }")
        .def_readonly("iter", &smurff::StatusItem::iter, "Current iteration in current phase")
        .def_readonly("rmse_avg", &smurff::StatusItem::rmse_avg, "Averag RMSE for test matrix across all samples")
        .def_readonly("rmse_1sample", &smurff::StatusItem::rmse_1sample, "RMSE for test matrix of last sample" )
        .def_readonly("train_rmse", &smurff::StatusItem::train_rmse, "RMSE for train matrix of last sample" )
        .def_readonly("auc_avg", &smurff::StatusItem::auc_avg, "Average ROC AUC of the test matrix across all samples"
                                                               "Only available if you provided a threshold")
        .def_readonly("auc_1sample", &smurff::StatusItem::auc_1sample, "ROC AUC of the test matrix of the last sample"
                                                               "Only available if you provided a threshold")
        .def_readonly("elapsed_iter", &smurff::StatusItem::elapsed_iter, "Number of seconds the last sampling iteration took")
        .def_readonly("nnz_per_sec", &smurff::StatusItem::nnz_per_sec, "Compute performance indicator; number of non-zero elements in train processed per second")
        .def_readonly("samples_per_sec", &smurff::StatusItem::samples_per_sec, "Compute performance indicator; number of rows and columns in U/V processed per second")
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
