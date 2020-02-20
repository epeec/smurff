#include <cstdio>
#include <fstream>

#include <boost/filesystem/operations.hpp>

#include "catch.hpp"

#include <SmurffCpp/Types.h>

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Predict/PredictSession.h>
#include <SmurffCpp/Sessions/SessionFactory.h>
#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/OutputFile.h>
#include <SmurffCpp/result.h>

#include "Tests.h"

#ifdef USE_BOOST_RANDOM
#define TAG_MATRIX_TESTS "[matrix][random]"
#else
#define TAG_MATRIX_TESTS "[matrix][random][!mayfail]"
#endif

namespace fs = boost::filesystem;

namespace smurff {
namespace test {

static Config& prepareResultDir(Config &config, const std::string &dir)
{
  fs::path output_dir("tests_output");
  fs::create_directory(output_dir);

  std::string save_dir(dir);
  save_dir.erase(std::remove_if(save_dir.begin(), save_dir.end(),
                           [](char c) {
                              const std::string special_chars("\\/:*\"<>|");
                              return special_chars.find(c) != std::string::npos;
                           }
            ), save_dir.end());
  
  output_dir /= fs::path(save_dir);
 
  config.setSaveFreq(1);
  config.setSavePrefix(output_dir.native());
  fs::remove_all(output_dir);
  fs::create_directory(output_dir);
  return config;
}

TEST_CASE("PredictSession/BPMF")
{
  Config config = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal});
  prepareResultDir(config, Catch::getResultCapture().getCurrentTestName() + "_train");

  std::shared_ptr<ISession> session = SessionFactory::create_session(config);
  session->run();

  // std::cout << "Prediction from Session RMSE: " << session->getRmseAvg() <<
  // std::endl;

  std::string root_fname = session->getOutputFile()->getFullPath();
  auto rf = std::make_shared<OutputFile>(root_fname);

  {
    prepareResultDir(config, Catch::getResultCapture().getCurrentTestName() + "_predict");
    PredictSession s(rf, config);

    // test predict from TensorConfig
    auto result = s.predict(config.getTest());

    // std::cout << "Prediction from OutputFile RMSE: " << result->rmse_avg <<
    // std::endl;
    REQUIRE(session->getRmseAvg() == Approx(result->rmse_avg).epsilon(APPROX_EPSILON));
  }

  {
    PredictSession s(rf, config);
    s.run();
    auto result = s.getResult();

    // std::cout << "Prediction from OutputFile+Config RMSE: " << result->rmse_avg
    // << std::endl;
    REQUIRE(session->getRmseAvg() == Approx(result->rmse_avg).epsilon(APPROX_EPSILON));
  }
}

//=================================================================

TEST_CASE("PredictSession/Features/1", TAG_MATRIX_TESTS) {
  std::shared_ptr<SideInfoConfig> rowSideInfoDenseMatrixConfig = makeSideInfoConfig(rowSideDenseMatrix);

  Config config = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::normal})
                      .addSideInfoConfig(0, rowSideInfoDenseMatrixConfig);
  prepareResultDir(config, Catch::getResultCapture().getCurrentTestName());

  std::shared_ptr<ISession> session = SessionFactory::create_session(config);
  session->run();

  PredictSession predict_session(session->getOutputFile());

  auto sideInfoMatrix = matrix_utils::dense_to_eigen(*rowSideInfoDenseMatrixConfig->getSideInfo());
  auto trainMatrix = smurff::matrix_utils::dense_to_eigen(trainDenseMatrix);

#if 0
    std::cout << "sideInfo =\n" << sideInfoMatrix << std::endl;
    std::cout << "train    =\n" << trainMatrix << std::endl;
#endif

  for (int r = 0; r < sideInfoMatrix.rows(); r++) {
#if 0
        std::cout << "=== row " << r << " ===\n";
#endif

    auto predictions = predict_session.predict(0, sideInfoMatrix.row(r));
#if 0
        int i = 0;
        for (auto P : predictions)
        {
            std::cout << "p[" << i++ << "] = " << P->transpose() << std::endl;
        }
#endif
  }
}

TEST_CASE("PredictSession/Features/2", TAG_MATRIX_TESTS) {
  /*
       BetaPrecision: 1.00
  U = np.array([ [ 1, 2, -1, -2  ] ])
  V = np.array([ [ 2, 2, 1, 2 ] ])
  U*V =
    [[ 2,  2,  1,  2],
     [ 4,  4,  2,  4],
     [-2, -2, -1, -2],
     [-4, -4, -2, -4]])
  */

  NoiseConfig noise_cfg(NoiseTypes::fixed);
  noise_cfg.setPrecision(1.);
  MatrixConfig trainMatrixConfig;
  {
    std::vector<std::uint32_t> trainMatrixConfigRows = {0, 0, 1, 1, 2, 2};
    std::vector<std::uint32_t> trainMatrixConfigCols = {0, 1, 2, 3, 0, 1};
    std::vector<double> trainMatrixConfigVals = {2, 2, 2, 4, -2, -2};

    trainMatrixConfig =
        MatrixConfig(4, 4, trainMatrixConfigRows, trainMatrixConfigCols, trainMatrixConfigVals, noise_cfg, true);
  }

  MatrixConfig testMatrixConfig;
  {
    std::vector<std::uint32_t> testMatrixConfigRows = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    std::vector<std::uint32_t> testMatrixConfigCols = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> testMatrixConfigVals = {2, 2, 1, 2, 4, 4, 2, 4, -2, -2, -1, -2, -4, -4, -2, -4};
    testMatrixConfig =
        MatrixConfig(4, 4, testMatrixConfigRows, testMatrixConfigCols, testMatrixConfigVals, noise_cfg, true);
  }

  std::shared_ptr<SideInfoConfig> rowSideInfoConfig;
  {
    NoiseConfig nc(NoiseTypes::sampled);
    nc.setPrecision(10.0);

    std::vector<std::uint32_t> rowSideInfoSparseMatrixConfigRows = {0, 1, 2, 3};
    std::vector<std::uint32_t> rowSideInfoSparseMatrixConfigCols = {0, 0, 0, 0};
    std::vector<double> rowSideInfoSparseMatrixConfigVals = {2, 4, -2, -4};

    auto mcfg =
        std::make_shared<MatrixConfig>(4, 1, rowSideInfoSparseMatrixConfigRows, rowSideInfoSparseMatrixConfigCols,
                                       rowSideInfoSparseMatrixConfigVals, nc, true);

    rowSideInfoConfig = std::make_shared<SideInfoConfig>();
    rowSideInfoConfig->setSideInfo(mcfg);
    rowSideInfoConfig->setDirect(true);
  }
  Config config = genConfig(trainMatrixConfig, testMatrixConfig, {PriorTypes::macau, PriorTypes::normal})
                      .addSideInfoConfig(0, rowSideInfoConfig);
  prepareResultDir(config, Catch::getResultCapture().getCurrentTestName());

  std::shared_ptr<ISession> session = SessionFactory::create_session(config);
  session->run();

  PredictSession predict_session_in(session->getOutputFile());
  auto in_matrix_predictions = predict_session_in.predict(config.getTest())->m_predictions;

  PredictSession predict_session_out(session->getOutputFile());
  auto sideInfoMatrix = matrix_utils::sparse_to_eigen(*rowSideInfoConfig->getSideInfo());
  int d = config.getTrain()->getDims()[0];
  for (int r = 0; r < d; r++) {
    auto feat = sideInfoMatrix.row(r).transpose();
    auto out_of_matrix_predictions = predict_session_out.predict(0, feat);
    // Vector out_of_matrix_averages =
    // out_of_matrix_predictions->colwise().mean();

#undef DEBUG_OOM_PREDICT
#ifdef DEBUG_OOM_PREDICT
    for (auto p : in_matrix_predictions) {
      if (p.coords[0] == r) {
        std::cout << "in: " << p << std::endl;
        std::cout << "  out: " << out_of_matrix_averages.row(p.coords[1]) << std::endl;
      }
    }
#endif
  }
}

} // namespace test
} // namespace smurff
