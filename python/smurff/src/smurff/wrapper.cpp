#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>


#include <SmurffCpp/Types.h>
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
        .def("setPriorTypes", py::overload_cast<std::vector<std::string>>(&smurff::Config::setPriorTypes))
        .def("setModelInitType", py::overload_cast<std::string>(&smurff::Config::setModelInitType))
        .def("setSaveName", &smurff::Config::setSaveName)
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

    py::class_<smurff::ResultItem>(m, "ResultItem", "Predictions for a single point in the matrix/tensor")
        .def("__str__", &smurff::ResultItem::to_string)
        .def_property_readonly("coords",  [](const smurff::ResultItem &r) { return r.coords.as_vector(); })
        .def_readonly("val", &smurff::ResultItem::val)
        .def_readonly("pred_1sample", &smurff::ResultItem::pred_1sample)
        .def_readonly("pred_avg", &smurff::ResultItem::pred_avg)
        .def_readonly("var", &smurff::ResultItem::var)
        .def_readonly("nsamples", &smurff::ResultItem::nsamples)
        .def_readonly("pred_all", &smurff::ResultItem::pred_all)
        ;

    py::class_<smurff::SparseTensor>(m, "SparseTensor")
        .def(py::init<
          const std::vector<std::uint64_t> &,
          const std::vector<std::vector<std::uint32_t>> &,
          const std::vector<double> &
        >())
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
        .def(py::init<const smurff::Config &>())
        .def("__str__", &smurff::ISession::infoAsString)

        // add data
        .def("setTrain", &smurff::PythonSession::setTrainDense<smurff::Matrix>)
        .def("setTrain", &smurff::PythonSession::setTrainSparse<smurff::SparseMatrix>)
        .def("setTrain", &smurff::PythonSession::setTrainDense<smurff::DenseTensor>)
        .def("setTrain", &smurff::PythonSession::setTrainSparse<smurff::SparseTensor>)

        .def("setTest", &smurff::PythonSession::setTest<smurff::SparseMatrix>)
        .def("setTest", &smurff::PythonSession::setTest<smurff::SparseTensor>)
        
        .def("addSideInfo", &smurff::PythonSession::addSideInfoDense)
        .def("addSideInfo", &smurff::PythonSession::addSideInfoSparse)
        
        .def("addData", &smurff::PythonSession::addDataDense<smurff::Matrix>)
        .def("addData", &smurff::PythonSession::addDataSparse<smurff::SparseMatrix>)
        .def("addData", &smurff::PythonSession::addDataDense<smurff::DenseTensor>)
        .def("addData", &smurff::PythonSession::addDataSparse<smurff::SparseTensor>)

        .def("addPropagatedPosterior", &smurff::PythonSession::addPropagatedPosterior)

        // get result functions
        .def("getStatus", &smurff::TrainSession::getStatus)
        .def("getRmseAvg", &smurff::TrainSession::getRmseAvg)
        .def("getTestPredictions", [](const smurff::PythonSession &s) { return s.getResult().m_predictions; })

        // run functions
        .def("init", &smurff::TrainSession::init)
        .def("step", &smurff::PythonSession::step)
        .def("interrupted", &smurff::PythonSession::interrupted)
        ;
}
