#include  <fstream>

#include "catch.hpp"

#include <SmurffCpp/Types.h>

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Sessions/SessionFactory.h>
#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/RootFile.h>
#include <SmurffCpp/Predict/PredictSession.h>
#include <SmurffCpp/result.h>

/////////////////////////////////////////////////////////////////////////////////////////////////
// Code for printing test results that can then be copy-pasted into tests as expected results
/////////////////////////////////////////////////////////////////////////////////////////////////
//

void printActualResults(int nr, double actualRmseAvg, const std::vector<smurff::ResultItem>& actualResults)
{
   std::ofstream os("TestsSmurff_ExpectedResults.h", std::ofstream::app);

   os << "{ " << nr << ",\n"
      << "  { " << std::fixed << std::setprecision(16) << actualRmseAvg << "," << std::endl
      << "      {\n";

   for (const auto &actualResultItem : actualResults) 
   {
      os << std::setprecision(16);
      os << "         { { " << actualResultItem.coords << " }, "
         << actualResultItem.val << ", "
         << std::fixed << actualResultItem.pred_1sample << ", "
         << actualResultItem.pred_avg << ", "
         << actualResultItem.var << ", "
         << " }," << std::endl;
    }

    os << "      }\n"
       << "  }\n"
       << "},\n";
}


//#define PRINT_ACTUAL_RESULTS(nr)
#define PRINT_ACTUAL_RESULTS(nr) printActualResults(nr, actualRmseAvg, actualResults);

using namespace smurff;

struct ExpectedResult {
   double rmseAvg;
   std::vector<ResultItem> resultItems;
};
std::map<int, ExpectedResult> expectedResults = {
#include "TestsSmurff_ExpectedResults.h"
};

static NoiseConfig fixed_ncfg(NoiseTypes::fixed);

// dense train data (matrix/tensor 2d/tensor 3d)

#ifdef USE_BOOST_RANDOM
#define TAG_MATRIX_TESTS "[matrix][random]"
#define TAG_TWO_DIMENTIONAL_TENSOR_TESTS "[tensor2d][random]"
#define TAG_THREE_DIMENTIONAL_TENSOR_TESTS "[tensor3d][random]"
#define TAG_VS_TESTS "[versus][random]"
#else
#define TAG_MATRIX_TESTS "[matrix][random][!mayfail]"
#define TAG_TWO_DIMENTIONAL_TENSOR_TESTS "[tensor2d][random][!mayfail]"
#define TAG_THREE_DIMENTIONAL_TENSOR_TESTS "[tensor3d][random][!mayfail]"
#define TAG_VS_TESTS "[versus][random][!mayfail]"
#endif

Config getTestsSmurffConfig()
{
   Config config;
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setNumThreads(1);
   config.setNumLatent(4);
   return config;
}

std::shared_ptr<MatrixConfig> getTrainDenseMatrixConfig()
{
   std::vector<double> trainMatrixConfigVals = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   std::shared_ptr<MatrixConfig> trainMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, trainMatrixConfigVals, fixed_ncfg);
   return trainMatrixConfig;
}

std::shared_ptr<TensorConfig> getTrainDenseTensor2dConfig()
{
   std::vector<double> trainTensorConfigVals = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   std::shared_ptr<TensorConfig> trainTensorConfig =
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 3, 4 }), trainTensorConfigVals.data(), fixed_ncfg);
   return trainTensorConfig;
}

std::shared_ptr<TensorConfig> getTrainDenseTensor3dConfig()
{
   std::vector<double> trainTensorConfigVals = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
   std::shared_ptr<TensorConfig> trainTensorConfig =
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 2, 3, 4 }), trainTensorConfigVals, fixed_ncfg);
   return trainTensorConfig;
}

// sparse train data (matrix/tensor 2d)

std::shared_ptr<MatrixConfig> getTrainSparseMatrixConfig()
{
   std::vector<std::uint32_t> trainMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> trainMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> trainMatrixConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> trainMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, trainMatrixConfigRows, trainMatrixConfigCols, trainMatrixConfigVals, fixed_ncfg, true);
   return trainMatrixConfig;
}

std::shared_ptr<TensorConfig> getTrainSparseTensor2dConfig()
{
   std::vector<std::vector<std::uint32_t>> trainTensorConfigCols =
      {
        { 0, 0, 0, 0, 2, 2, 2, 2 },
        { 0, 1, 2, 3, 0, 1, 2, 3 }
      };
   std::vector<double> trainTensorConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<TensorConfig> trainTensorConfig =
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 3, 4 }), trainTensorConfigCols, trainTensorConfigVals, fixed_ncfg, true);
   return trainTensorConfig;
}

// sparse test data (matrix/tensor 2d/tensor 3d)

std::shared_ptr<MatrixConfig> getTestSparseMatrixConfig()
{
   std::vector<std::uint32_t> testMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2};
   std::vector<std::uint32_t> testMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> testMatrixConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> testMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, testMatrixConfigRows, testMatrixConfigCols, testMatrixConfigVals, fixed_ncfg, true);
   return testMatrixConfig;
}

std::shared_ptr<TensorConfig> getTestSparseTensor2dConfig()
{
    std::vector<std::vector<std::uint32_t>> testTensorConfigCols =
      {
         { 0, 0, 0, 0, 2, 2, 2, 2 },
         { 0, 1, 2, 3, 0, 1, 2, 3 }
      };
   std::vector<double> testTensorConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<TensorConfig> testTensorConfig =
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 3, 4 }), testTensorConfigCols, testTensorConfigVals, fixed_ncfg, true);
   return testTensorConfig;
}

std::shared_ptr<TensorConfig> getTestSparseTensor3dConfig()
{
   std::vector<std::vector<std::uint32_t>> testTensorConfigCols =
      {
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 2, 2, 2, 2 },
        { 0, 1, 2, 3, 0, 1, 2, 3 }
      };
   std::vector<double> testTensorConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<TensorConfig> testTensorConfig =
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 2, 3, 4 }), testTensorConfigCols, testTensorConfigVals, fixed_ncfg, true);
   return testTensorConfig;
}

// aux data

std::shared_ptr<MatrixConfig> getRowAuxDataDenseMatrixConfig()
{
   std::vector<double> rowAuxDataDenseMatrixConfigVals = { 1, 2, 3 };
   std::shared_ptr<MatrixConfig> rowAuxDataDenseMatrixConfig =
      std::make_shared<MatrixConfig>(3, 1, rowAuxDataDenseMatrixConfigVals, fixed_ncfg);
   rowAuxDataDenseMatrixConfig->setPos(PVec<>({0,1}));
   return rowAuxDataDenseMatrixConfig;
}

std::shared_ptr<MatrixConfig> getColAuxDataDenseMatrixConfig()
{
   std::vector<double> colAuxDataDenseMatrixConfigVals = { 1, 2, 3, 4 };
   std::shared_ptr<MatrixConfig> colAuxDataDenseMatrixConfig =
      std::make_shared<MatrixConfig>(1, 4, colAuxDataDenseMatrixConfigVals, fixed_ncfg);
   colAuxDataDenseMatrixConfig->setPos(PVec<>({1,0}));
   return colAuxDataDenseMatrixConfig;
}

// side info

std::shared_ptr<MatrixConfig> getRowSideInfoDenseMatrixConfig()
{
   NoiseConfig nc(NoiseTypes::sampled);
   nc.setPrecision(10.0);

   std::vector<double> rowSideInfoDenseMatrixConfigVals = { 1, 2, 3 };
   std::shared_ptr<MatrixConfig> rowSideInfoDenseMatrixConfig =
      std::make_shared<MatrixConfig>(3, 1, rowSideInfoDenseMatrixConfigVals, nc);
   return rowSideInfoDenseMatrixConfig;
}

std::shared_ptr<MatrixConfig> getColSideInfoDenseMatrixConfig()
{
   NoiseConfig nc(NoiseTypes::sampled);
   nc.setPrecision(10.0);

   std::vector<double> colSideInfoDenseMatrixConfigVals = { 1, 2, 3, 4 };
   std::shared_ptr<MatrixConfig> colSideInfoDenseMatrixConfig =
      std::make_shared<MatrixConfig>(4, 1, colSideInfoDenseMatrixConfigVals, nc);
   return colSideInfoDenseMatrixConfig;
}

std::shared_ptr<MatrixConfig> getRowSideInfoSparseMatrixConfig()
{
   NoiseConfig nc(NoiseTypes::sampled);
   nc.setPrecision(10.0);

   std::vector<std::uint32_t> rowSideInfoSparseMatrixConfigRows = {0, 1, 2};
   std::vector<std::uint32_t> rowSideInfoSparseMatrixConfigCols = {0, 0, 0};
   std::vector<double> rowSideInfoSparseMatrixConfigVals = { 1, 2, 3 };
   std::shared_ptr<MatrixConfig> rowSideInfoSparseMatrixConfig =
      std::make_shared<MatrixConfig>(3, 1, rowSideInfoSparseMatrixConfigRows, rowSideInfoSparseMatrixConfigCols, rowSideInfoSparseMatrixConfigVals, nc, true);
   return rowSideInfoSparseMatrixConfig;
}

std::shared_ptr<MatrixConfig> getColSideInfoSparseMatrixConfig()
{
   NoiseConfig nc(NoiseTypes::sampled);
   nc.setPrecision(10.0);

   std::vector<std::uint32_t> colSideInfoSparseMatrixConfigRows = {0, 1, 2, 3};
   std::vector<std::uint32_t> colSideInfoSparseMatrixConfigCols = {0, 0, 0, 0};
   std::vector<double> colSideInfoSparseMatrixConfigVals = { 1, 2, 3, 4 };
   std::shared_ptr<MatrixConfig> colSideInfoSparseMatrixConfig =
      std::make_shared<MatrixConfig>(4, 1, colSideInfoSparseMatrixConfigRows, colSideInfoSparseMatrixConfigCols, colSideInfoSparseMatrixConfigVals, nc, true);
   return colSideInfoSparseMatrixConfig;
}

std::shared_ptr<MatrixConfig> getRowSideInfoDenseMatrix3dConfig()
{
   NoiseConfig nc(NoiseTypes::sampled);
   nc.setPrecision(10.0);

   std::vector<double> rowSideInfoDenseMatrixConfigVals = { 1, 2, 3, 4, 5, 6 };
   std::shared_ptr<MatrixConfig> rowSideInfoDenseMatrixConfig =
      std::make_shared<MatrixConfig>(2, 3, rowSideInfoDenseMatrixConfigVals, nc);
   return rowSideInfoDenseMatrixConfig;
}


std::shared_ptr<SideInfoConfig> getRowSideInfoDenseConfig(bool direct = true, double tol = 1e-6)
{
   std::shared_ptr<MatrixConfig> mcfg = getRowSideInfoDenseMatrixConfig();
   
   std::shared_ptr<SideInfoConfig> picfg = std::make_shared<SideInfoConfig>();
   picfg->setSideInfo(mcfg);
   picfg->setDirect(direct);
   picfg->setTol(tol);

   return picfg;
}

std::shared_ptr<SideInfoConfig> getColSideInfoDenseConfig(bool direct = true, double tol = 1e-6)
{
   std::shared_ptr<MatrixConfig> mcfg = getColSideInfoDenseMatrixConfig();

   std::shared_ptr<SideInfoConfig> picfg = std::make_shared<SideInfoConfig>();
   picfg->setSideInfo(mcfg);
   picfg->setDirect(direct);
   picfg->setTol(tol);

   return picfg;
}

std::shared_ptr<SideInfoConfig> getRowSideInfoSparseConfig(bool direct = true, double tol = 1e-6)
{
   std::shared_ptr<MatrixConfig> mcfg = getRowSideInfoSparseMatrixConfig();

   std::shared_ptr<SideInfoConfig> picfg = std::make_shared<SideInfoConfig>();
   picfg->setSideInfo(mcfg);
   picfg->setDirect(direct);
   picfg->setTol(tol);

   return picfg;
}

std::shared_ptr<SideInfoConfig> getColSideInfoSparseConfig(bool direct = true, double tol = 1e-6)
{
   std::shared_ptr<MatrixConfig> mcfg = getColSideInfoSparseMatrixConfig();

   std::shared_ptr<SideInfoConfig> picfg = std::make_shared<SideInfoConfig>();
   picfg->setSideInfo(mcfg);
   picfg->setDirect(direct);
   picfg->setTol(tol);

   return picfg;
}

std::shared_ptr<SideInfoConfig> getRowSideInfoDenseMacauPrior3dConfig(bool direct = true, double tol = 1e-6)
{
   std::shared_ptr<MatrixConfig> mcfg = getRowSideInfoDenseMatrix3dConfig();

   std::shared_ptr<SideInfoConfig> picfg = std::make_shared<SideInfoConfig>();
   picfg->setSideInfo(mcfg);
   picfg->setDirect(direct);
   picfg->setTol(tol);

   return picfg;
}

//result comparison

void REQUIRE_RESULT_ITEMS(const std::vector<ResultItem>& actualResultItems, const std::vector<ResultItem>& expectedResultItems)
{
   REQUIRE(actualResultItems.size() == expectedResultItems.size());
   double single_item_epsilon = APPROX_EPSILON * 10;
   for (std::vector<ResultItem>::size_type i = 0; i < actualResultItems.size(); i++)
   {
      const ResultItem& actualResultItem = actualResultItems[i];
      const ResultItem& expectedResultItem = expectedResultItems[i];
      REQUIRE(actualResultItem.coords == expectedResultItem.coords);
      REQUIRE(actualResultItem.val == expectedResultItem.val);
      REQUIRE(actualResultItem.pred_1sample == Approx(expectedResultItem.pred_1sample).epsilon(single_item_epsilon));
      REQUIRE(actualResultItem.pred_avg == Approx(expectedResultItem.pred_avg).epsilon(single_item_epsilon));
      REQUIRE(actualResultItem.var == Approx(expectedResultItem.var).epsilon(single_item_epsilon));
   }
}

void runSession(Config &config, int nr)
{
   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   const std::vector<ResultItem> & actualResults = session->getResultItems();

   PRINT_ACTUAL_RESULTS(nr)
   double &expectedRmseAvg = expectedResults[nr].rmseAvg;
   auto &expectedResultItems = expectedResults[nr].resultItems;

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(actualResults, expectedResultItems);
}

void compareSessions(Config &matrixSessionConfig, Config &tensorSessionConfig)
{
   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(matrixSession->getResultItems(), tensorSession->getResultItems());
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: normal normal
//   aux-data: none none
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior normal normal --aux-data none none"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::normal, PriorTypes::normal});
   runSession(config, 359);
}

//
//      train: sparse matrix
//       test: sparse matrix
//     priors: normal normal
//   aux-data: none none
//
TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior normal normal --aux-data none none"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainSparseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::normal, PriorTypes::normal});
   runSession(config, 411);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: normal normal
//   aux-data: dense_matrix dense_matrix
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior normal normal --aux-data <dense_matrix> <dense_matrix>"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<TensorConfig> rowAuxDataDenseMatrixConfig = getRowAuxDataDenseMatrixConfig();
   std::shared_ptr<TensorConfig> colAuxDataDenseMatrixConfig = getColAuxDataDenseMatrixConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::normal, PriorTypes::normal});
   config.addAuxData({ rowAuxDataDenseMatrixConfig });
   config.addAuxData({ colAuxDataDenseMatrixConfig });
   runSession(config, 467);
}

//
//      train: sparse matrix
//       test: sparse matrix
//     priors: normal normal
//   aux-data: dense_matrix dense_matrix
//
TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior normal normal --aux-data <dense_matrix> <dense_matrix>"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<TensorConfig> rowAuxDataDenseMatrixConfig = getRowAuxDataDenseMatrixConfig();
   std::shared_ptr<TensorConfig> colAuxDataDenseMatrixConfig = getColAuxDataDenseMatrixConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainSparseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::normal, PriorTypes::normal});
   config.addAuxData({ rowAuxDataDenseMatrixConfig });
   config.addAuxData({ colAuxDataDenseMatrixConfig });
   runSession(config, 523);
}

//=================================================================

//
//      train: dense matrix
//       test: sparse matrix
//     priors: spikeandslab spikeandslab
//   aux-data: none none
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior spikeandslab spikeandslab --aux-data none none"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::spikeandslab});
   runSession(config, 577);
}

//
//      train: sparse matrix
//       test: sparse matrix
//     priors: spikeandslab spikeandslab
//   aux-data: none none
//
TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior spikeandslab spikeandslab --aux-data none none"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainSparseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::spikeandslab});
   runSession(config, 629);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: spikeandslab spikeandslab
//   aux-data: dense_matrix dense_matrix
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior spikeandslab spikeandslab --aux-data <dense_matrix> <dense_matrix>"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<TensorConfig> rowAuxDataDenseMatrixConfig = getRowAuxDataDenseMatrixConfig();
   std::shared_ptr<TensorConfig> colAuxDataDenseMatrixConfig = getColAuxDataDenseMatrixConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::spikeandslab});
   config.addAuxData({ rowAuxDataDenseMatrixConfig });
   config.addAuxData({ colAuxDataDenseMatrixConfig });
   runSession(config, 685);
}

//
//      train: sparse matrix
//       test: sparse matrix
//     priors: spikeandslab spikeandslab
//   aux-data: dense_matrix dense_matrix
//
TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior spikeandslab spikeandslab --aux-data <dense_matrix> <dense_matrix>"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<TensorConfig> rowAuxDataDenseMatrixConfig = getRowAuxDataDenseMatrixConfig();
   std::shared_ptr<TensorConfig> colAuxDataDenseMatrixConfig = getColAuxDataDenseMatrixConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainSparseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::spikeandslab});
   config.addAuxData({ rowAuxDataDenseMatrixConfig });
   config.addAuxData({ colAuxDataDenseMatrixConfig });
   runSession(config, 741);
}

//=================================================================

//
//      train: dense matrix
//       test: sparse matrix
//     priors: normalone normalone
//   aux-data: none none
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior normalone normalone --aux-data none none"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::normalone, PriorTypes::normalone});
   runSession(config, 795);
}

//
//      train: sparse matrix
//       test: sparse matrix
//     priors: normalone normalone
//   aux-data: none none
//
TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior normalone normalone --aux-data none none"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainSparseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::normalone, PriorTypes::normalone});
   runSession(config, 847);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: normalone normalone
//   aux-data: dense_matrix dense_matrix
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior normalone normalone --aux-data <dense_matrix> <dense_matrix>"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<TensorConfig> rowAuxDataDenseMatrixConfig = getRowAuxDataDenseMatrixConfig();
   std::shared_ptr<TensorConfig> colAuxDataDenseMatrixConfig = getColAuxDataDenseMatrixConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::normalone, PriorTypes::normalone});
   config.addAuxData({ rowAuxDataDenseMatrixConfig });
   config.addAuxData({ colAuxDataDenseMatrixConfig });
   runSession(config, 903);
}

//
//      train: sparse matrix
//       test: sparse matrix
//     priors: normalone normalone
//   aux-data: dense_matrix dense_matrix
//
TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior normalone normalone --aux-data <dense_matrix> <dense_matrix>"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<TensorConfig> rowAuxDataDenseMatrixConfig = getRowAuxDataDenseMatrixConfig();
   std::shared_ptr<TensorConfig> colAuxDataDenseMatrixConfig = getColAuxDataDenseMatrixConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainSparseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::normalone, PriorTypes::normalone});
   config.addAuxData({ rowAuxDataDenseMatrixConfig });
   config.addAuxData({ colAuxDataDenseMatrixConfig });
   runSession(config, 959);
}

//=================================================================

//
//      train: dense matrix
//       test: sparse matrix
//     priors: macau macau
//   features: row_side_info_dense_matrix col_side_info_dense_matrix
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior macau macau --aux-data <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<SideInfoConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseConfig();
   std::shared_ptr<SideInfoConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::macau, PriorTypes::macau});
   config.addSideInfoConfig(0, rowSideInfoDenseMatrixConfig);
   config.addSideInfoConfig(1, colSideInfoDenseMatrixConfig);
   runSession(config, 1018);
}

//
//      train: sparse matrix
//       test: sparse matrix
//     priors: macau macau
//   features: row_side_info_dense_matrix col_side_info_dense_matrix
//     direct: true
//
TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior macau macau --aux-data <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<SideInfoConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseConfig();
   std::shared_ptr<SideInfoConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainSparseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::macau, PriorTypes::macau});
   config.addSideInfoConfig(0, rowSideInfoDenseMatrixConfig);
   config.addSideInfoConfig(1, colSideInfoDenseMatrixConfig);
   runSession(config, 1075);
}

//=================================================================

//
//      train: dense matrix
//       test: sparse matrix
//     priors: macauone macauone
//   features: row_side_info_sparse_matrix col_side_info_sparse_matrix
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior macauone macauone --aux-data <row_side_info_sparse_matrix> <col_side_info_sparse_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   std::shared_ptr<SideInfoConfig> rowSideInfoSparseMatrixConfig = getRowSideInfoSparseConfig();
   std::shared_ptr<SideInfoConfig> colSideInfoSparseMatrixConfig = getColSideInfoSparseConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::macauone, PriorTypes::macauone});
   config.addSideInfoConfig(0, rowSideInfoSparseMatrixConfig);
   config.addSideInfoConfig(1, colSideInfoSparseMatrixConfig);
   runSession(config, 1135);
}

//
//      train: sparse matrix
//       test: sparse matrix
//     priors: macauone macauone
//   features: row_side_info_sparse_matrix col_side_info_sparse_matrix
//     direct: true
//
TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior macauone macauone --aux-data <row_side_info_sparse_matrix> <col_side_info_sparse_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   std::shared_ptr<SideInfoConfig> rowSideInfoSparseMatrixConfig = getRowSideInfoSparseConfig();
   std::shared_ptr<SideInfoConfig> colSideInfoSparseMatrixConfig = getColSideInfoSparseConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainSparseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::macauone, PriorTypes::macauone});
   config.addSideInfoConfig(0, rowSideInfoSparseMatrixConfig);
   config.addSideInfoConfig(1, colSideInfoSparseMatrixConfig);
   runSession(config, 1193);
}

//=================================================================

//
//      train: dense matrix
//       test: sparse matrix
//     priors: macau normal
//   features: row_side_info_dense_matrix none
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior macau normal --aux-data <row_side_info_dense_matrix> none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<SideInfoConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::macau, PriorTypes::normal});
   config.addSideInfoConfig(0, rowSideInfoDenseMatrixConfig);
   runSession(config, 1250);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: normal macau
//   features: none col_side_info_dense_matrix
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior normal macau --aux-data none <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<SideInfoConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::normal, PriorTypes::macau});
   config.addSideInfoConfig(1, colSideInfoDenseMatrixConfig);
   runSession(config, 1305);
}

//test throw - normal prior should not have side info

//
//      train: dense matrix
//       test: sparse matrix
//     priors: macau normal
//   features: col_side_info_dense_matrix row_side_info_dense_matrix
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior macau normal --aux-data <col_side_info_dense_matrix> <row_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<SideInfoConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseConfig();
   std::shared_ptr<SideInfoConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.addSideInfoConfig(1, colSideInfoDenseMatrixConfig);
   config.addSideInfoConfig(0, rowSideInfoDenseMatrixConfig);
   config.setPriorTypes({PriorTypes::macau, PriorTypes::normal});

   REQUIRE_THROWS(SessionFactory::create_session(config));
}

//test throw - macau prior should have side info

//
//      train: dense matrix
//       test: sparse matrix
//     priors: macau normal
//   features: none none
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior macau normal --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<SideInfoConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.addSideInfoConfig(1, rowSideInfoDenseMatrixConfig); // added to wrong mode
   config.setPriorTypes({PriorTypes::macau, PriorTypes::normal});

   REQUIRE_THROWS(SessionFactory::create_session(config));
}

//test throw - wrong dimentions of side info

//
//      train: dense matrix
//       test: sparse matrix
//     priors: macau normal
//   features: col_side_info_dense_matrix none
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior macau normal --aux-data <col_side_info_dense_matrix> none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<SideInfoConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::macau, PriorTypes::normal});
   config.addSideInfoConfig(1, colSideInfoDenseMatrixConfig);

   REQUIRE_THROWS(SessionFactory::create_session(config));
}

//=================================================================

//
//      train: dense matrix
//       test: sparse matrix
//     priors: normal spikeandslab
//   aux-data: none none
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior normal spikeandslab --aux-data none none"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::normal, PriorTypes::spikeandslab});
   runSession(config, 1466);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: spikeandslab normal
//   aux-data: none none
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior spikeandslab normal --aux-data none none"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::normal});
   runSession(config, 1518);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: normal spikeandslab
//   aux-data: none dense_matrix
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior normal spikeandslab --aux-data none <dense_matrix>"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<TensorConfig> colAuxDataDenseMatrixConfig = getColAuxDataDenseMatrixConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::normal});
   config.addAuxData({ colAuxDataDenseMatrixConfig });
   runSession(config, 1572);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: spikeandslab normal
//   aux-data: dense_matrix none
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior spikeandslab normal --aux-data <dense_matrix> none"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<TensorConfig> rowAuxDataDenseMatrixConfig = getRowAuxDataDenseMatrixConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::normal});
   config.addAuxData({ rowAuxDataDenseMatrixConfig });
   runSession(config, 1626);
}

//=================================================================

//
//      train: dense matrix
//       test: sparse matrix
//     priors: macau spikeandslab
//   features: row_side_info_dense_matrix none
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior macau spikeandslab --aux-data <row_side_info_dense_matrix> none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<SideInfoConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::macau, PriorTypes::spikeandslab});
   config.addSideInfoConfig(0, rowSideInfoDenseMatrixConfig);
   runSession(config, 1683);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: spikeandslab macau
//   features: none col_side_info_dense_matrix
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior spikeandslab macau --aux-data none <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   , TAG_MATRIX_TESTS)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<SideInfoConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::macau});
   config.addSideInfoConfig(1, colSideInfoDenseMatrixConfig);
   runSession(config, 1738);
}

//=================================================================

//
//      train: dense 2D-tensor (matrix)
//       test: sparse 2D-tensor (matrix)
//     priors: normal normal
//   aux-data: none none
//
TEST_CASE("--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior normal normal --aux-data none none"
   , TAG_TWO_DIMENTIONAL_TENSOR_TESTS)
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainSparseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.setPriorTypes({PriorTypes::normal, PriorTypes::normal});
   runSession(config, 1792);
}

//
//      train: sparse 2D-tensor (matrix)
//       test: sparse 2D-tensor (matrix)
//     priors: normal normal
//   aux-data: none none
//
TEST_CASE("--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior normal normal --aux-data none none"
   , TAG_TWO_DIMENTIONAL_TENSOR_TESTS)
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainSparseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.setPriorTypes({PriorTypes::normal, PriorTypes::normal});
   runSession(config, 1844);
}

//=================================================================
//
//      train: dense 2D-tensor (matrix)
//       test: sparse 2D-tensor (matrix)
//     priors: spikeandslab spikeandslab
//   aux-data: none none
//
TEST_CASE("--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior spikeandslab spikeandslab --aux-data none none"
   , TAG_TWO_DIMENTIONAL_TENSOR_TESTS)
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainSparseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::spikeandslab});

   runSession(config, 1898);
}

//
//      train: sparse 2D-tensor (matrix)
//       test: sparse 2D-tensor (matrix)
//     priors: spikeandslab spikeandslab
//   aux-data: none none
//
TEST_CASE("--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior spikeandslab spikeandslab --aux-data none none"
   , TAG_TWO_DIMENTIONAL_TENSOR_TESTS)
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainSparseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::spikeandslab});
   runSession(config, 1950);
}

//=================================================================

//
//      train: dense 2D-tensor (matrix)
//       test: sparse 2D-tensor (matrix)
//     priors: normalone normalone
//   aux-data: none none
//
TEST_CASE("--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior normalone normalone --aux-data none none"
   , TAG_TWO_DIMENTIONAL_TENSOR_TESTS)
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainSparseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.setPriorTypes({PriorTypes::normalone, PriorTypes::normalone});
   runSession(config, 2004);
}

//
//      train: sparse 2D-tensor (matrix)
//       test: sparse 2D-tensor (matrix)
//     priors: normalone normalone
//   aux-data: none none
//
TEST_CASE("--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior normalone normalone --aux-data none none"
   , TAG_TWO_DIMENTIONAL_TENSOR_TESTS)
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainSparseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.setPriorTypes({PriorTypes::normalone, PriorTypes::normalone});
   runSession(config, 2056);
}

//=================================================================

//
//      train: dense 3D-tensor (matrix)
//       test: sparse 3D-tensor (matrix)
//     priors: normal normal normal
//   aux-data: none none
//
TEST_CASE("--train <train_dense_3d_tensor> --test <test_sparse_3d_tensor> --prior normal normal --aux-data none none"
   , TAG_THREE_DIMENTIONAL_TENSOR_TESTS)
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainDenseTensor3dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor3dConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainSparseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.setPriorTypes({PriorTypes::normal, PriorTypes::normal, PriorTypes::normal});
   runSession(config, 2110);
}

//=================================================================

//
//      train: dense 3D-tensor (matrix)
//       test: sparse 3D-tensor (matrix)
//     priors: spikeandslab spikeandslab
//   aux-data: none none
//
TEST_CASE("--train <train_dense_3d_tensor> --test <test_sparse_3d_tensor> --prior spikeandslab spikeandslab --aux-data none none"
   , TAG_THREE_DIMENTIONAL_TENSOR_TESTS)
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainDenseTensor3dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor3dConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainSparseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::spikeandslab, PriorTypes::spikeandslab});
   runSession(config, 2164);
}

//=================================================================

//not sure if this test produces correct results

//
//      train: dense 3D-tensor
//       test: sparse 3D-tensor
//     priors: macau normal
//   aux-data: row_dense_side_info none
//
TEST_CASE("--train <train_dense_3d_tensor> --test <test_sparse_3d_tensor> --prior macau normal --side-info row_dense_side_info none"
   , TAG_THREE_DIMENTIONAL_TENSOR_TESTS)
{
   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor3dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor3dConfig();
   std::shared_ptr<SideInfoConfig> rowSideInfoDenseMatrix3dConfig = getRowSideInfoDenseMacauPrior3dConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.setPriorTypes({PriorTypes::macau, PriorTypes::normal, PriorTypes::normal});
   config.addSideInfoConfig(0, rowSideInfoDenseMatrix3dConfig);
   runSession(config, 2222);
}

//=================================================================

//not sure if this test produces correct results

//
//      train: dense 3D-tensor
//       test: sparse 3D-tensor
//     priors: macauone normal
//   aux-data: row_dense_side_info none
//
TEST_CASE("--train <train_dense_3d_tensor> --test <test_sparse_3d_tensor> --prior macauone normal --side-info row_dense_side_info none"
   , TAG_THREE_DIMENTIONAL_TENSOR_TESTS"[!mayfail]")
{
   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor3dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor3dConfig();
   std::shared_ptr<SideInfoConfig> rowSideInfoDenseMatrix3dConfig = getRowSideInfoDenseMacauPrior3dConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.setPriorTypes({PriorTypes::macauone, PriorTypes::normal, PriorTypes::normal});
   config.addSideInfoConfig(0, rowSideInfoDenseMatrix3dConfig);

   runSession(config, 2280);
}

//=================================================================

//pairwise tests for 2d matrix vs 2d tensor
// normal normal
// normal spikeandslab
// spikeandslab normal
// spikeandslab spikeandslab

//
//      train: 1. dense 2D-tensor (matrix)
//             2. dense matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: normal normal
//   aux-data: none none
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_dense_matrix>    --test <test_sparse_matrix>    --prior normal normal --aux-data none none"
   "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior normal normal --aux-data none none"
   , TAG_VS_TESTS
)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig = getTestsSmurffConfig();
   matrixSessionConfig.setTrain(trainDenseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.setPriorTypes({PriorTypes::normal, PriorTypes::normal});

   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig = getTestsSmurffConfig();
   tensorSessionConfig.setTrain(trainDenseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.setPriorTypes({PriorTypes::normal, PriorTypes::normal});

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(matrixSession->getResultItems(), tensorSession->getResultItems());
}

//
//      train: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: normal normal
//   aux-data: none none
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_sparse_matrix>    --test <test_sparse_matrix>    --prior normal normal --aux-data none none"
   "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior normal normal --aux-data none none"
   , TAG_VS_TESTS
)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig = getTestsSmurffConfig();
   matrixSessionConfig.setTrain(trainSparseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.setPriorTypes({PriorTypes::normal, PriorTypes::normal});

   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig = getTestsSmurffConfig();
   tensorSessionConfig.setTrain(trainSparseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.setPriorTypes({PriorTypes::normal, PriorTypes::normal});

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(matrixSession->getResultItems(), tensorSession->getResultItems());
}

//
//      train: 1. dense 2D-tensor (matrix)
//             2. dense matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: normal spikeandslab
//   aux-data: none none
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_dense_matrix>    --test <test_sparse_matrix>    --prior normal spikeandslab --aux-data none none"
   "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior normal spikeandslab --aux-data none none"
   , TAG_VS_TESTS
)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig = getTestsSmurffConfig();
   matrixSessionConfig.setTrain(trainDenseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.setPriorTypes({PriorTypes::normal, PriorTypes::spikeandslab});

   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig = getTestsSmurffConfig();
   tensorSessionConfig.setTrain(trainDenseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.setPriorTypes({PriorTypes::normal, PriorTypes::spikeandslab});

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(matrixSession->getResultItems(), tensorSession->getResultItems());
}

//
//      train: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: normal spikeandslab
//   aux-data: none none
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_sparse_matrix>    --test <test_sparse_matrix>    --prior normal spikeandslab --aux-data none none"
   "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior normal spikeandslab --aux-data none none"
   , TAG_VS_TESTS
)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig = getTestsSmurffConfig();
   matrixSessionConfig.setTrain(trainSparseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.setPriorTypes({PriorTypes::normal, PriorTypes::spikeandslab});

   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig = getTestsSmurffConfig();
   tensorSessionConfig.setTrain(trainSparseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.setPriorTypes({PriorTypes::normal, PriorTypes::spikeandslab});

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(matrixSession->getResultItems(), tensorSession->getResultItems());
}

//
//      train: 1. dense 2D-tensor (matrix)
//             2. dense matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: spikeandslab normal
//   aux-data: none none
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_dense_matrix>    --test <test_sparse_matrix>    --prior spikeandslab normal --aux-data none none"
   "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior spikeandslab normal --aux-data none none"
   , TAG_VS_TESTS
)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig = getTestsSmurffConfig();
   matrixSessionConfig.setTrain(trainDenseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::normal});

   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig = getTestsSmurffConfig();
   tensorSessionConfig.setTrain(trainDenseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::normal});

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(matrixSession->getResultItems(), tensorSession->getResultItems());
}

//
//      train: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: spikeandslab normal
//   aux-data: none none
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_sparse_matrix>    --test <test_sparse_matrix>    --prior spikeandslab normal --aux-data none none"
   "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior spikeandslab normal --aux-data none none"
   , TAG_VS_TESTS
)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig = getTestsSmurffConfig();
   matrixSessionConfig.setTrain(trainSparseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::normal});

   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig = getTestsSmurffConfig();
   tensorSessionConfig.setTrain(trainSparseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::normal});

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(matrixSession->getResultItems(), tensorSession->getResultItems());
}

//
//      train: 1. dense 2D-tensor (matrix)
//             2. dense matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: spikeandslab spikeandslab
//   aux-data: none none
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_dense_matrix>    --test <test_sparse_matrix>    --prior spikeandslab spikeandslab --aux-data none none"
   "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior spikeandslab spikeandslab --aux-data none none"
   , TAG_VS_TESTS
)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig = getTestsSmurffConfig();
   matrixSessionConfig.setTrain(trainDenseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::spikeandslab});

   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig = getTestsSmurffConfig();
   tensorSessionConfig.setTrain(trainDenseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::spikeandslab});

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(matrixSession->getResultItems(), tensorSession->getResultItems());
}

//
//      train: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: spikeandslab spikeandslab
//   aux-data: none none
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_sparse_matrix>    --test <test_sparse_matrix>    --prior spikeandslab spikeandslab --aux-data none none"
   "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior spikeandslab spikeandslab --aux-data none none"
   , TAG_VS_TESTS
)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig = getTestsSmurffConfig();
   matrixSessionConfig.setTrain(trainSparseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::spikeandslab});

   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig = getTestsSmurffConfig();
   tensorSessionConfig.setTrain(trainSparseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::spikeandslab});

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(matrixSession->getResultItems(), tensorSession->getResultItems());
}

//==========================================================================

//
//      train: 1. dense 2D-tensor (matrix)
//             2. dense matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: normal normalone
//   aux-data: none none
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_dense_matrix>    --test <test_sparse_matrix>    --prior normal normalone --aux-data none none"
   "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior normal normalone --aux-data none none"
   , TAG_VS_TESTS
)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig = getTestsSmurffConfig();
   matrixSessionConfig.setTrain(trainDenseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.setPriorTypes({PriorTypes::normal, PriorTypes::normalone});

   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig = getTestsSmurffConfig();
   tensorSessionConfig.setTrain(trainDenseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.setPriorTypes({PriorTypes::normal, PriorTypes::normalone});

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(matrixSession->getResultItems(), tensorSession->getResultItems());
}

//
//      train: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: normal normalone
//   aux-data: none none
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_sparse_matrix>    --test <test_sparse_matrix>    --prior normal normalone --aux-data none none"
   "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior normal normalone --aux-data none none"
   , TAG_VS_TESTS
)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig = getTestsSmurffConfig();
   matrixSessionConfig.setTrain(trainSparseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.setPriorTypes({PriorTypes::normal, PriorTypes::normalone});

   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig = getTestsSmurffConfig();
   tensorSessionConfig.setTrain(trainSparseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.setPriorTypes({PriorTypes::normal, PriorTypes::normalone});

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(matrixSession->getResultItems(), tensorSession->getResultItems());
}

//
//      train: 1. dense 2D-tensor (matrix)
//             2. dense matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: normalone normal
//   aux-data: none none
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_dense_matrix>    --test <test_sparse_matrix>    --prior normalone normal --aux-data none none"
   "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior normalone normal --aux-data none none"
   , TAG_VS_TESTS
)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig = getTestsSmurffConfig();
   matrixSessionConfig.setTrain(trainDenseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.setPriorTypes({PriorTypes::normalone, PriorTypes::normal});

   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig = getTestsSmurffConfig();
   tensorSessionConfig.setTrain(trainDenseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.setPriorTypes({PriorTypes::normalone, PriorTypes::normal});

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(matrixSession->getResultItems(), tensorSession->getResultItems());
}

//
//      train: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: normalone normal
//   aux-data: none none
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_sparse_matrix>    --test <test_sparse_matrix>    --prior normalone normal --aux-data none none"
   "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior normalone normal --aux-data none none"
   , TAG_VS_TESTS
)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig = getTestsSmurffConfig();
   matrixSessionConfig.setTrain(trainSparseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.setPriorTypes({PriorTypes::normalone, PriorTypes::normal});

   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig = getTestsSmurffConfig();
   tensorSessionConfig.setTrain(trainSparseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.setPriorTypes({PriorTypes::normalone, PriorTypes::normal});

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(matrixSession->getResultItems(), tensorSession->getResultItems());
}

//
//      train: 1. dense 2D-tensor (matrix)
//             2. dense matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: normalone normalone
//   aux-data: none none
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_dense_matrix>    --test <test_sparse_matrix>    --prior normalone normalone --aux-data none none"
   "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior normalone normalone --aux-data none none"
   , TAG_VS_TESTS
)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig = getTestsSmurffConfig();
   matrixSessionConfig.setTrain(trainDenseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.setPriorTypes({PriorTypes::normalone, PriorTypes::normalone});

   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig = getTestsSmurffConfig();
   tensorSessionConfig.setTrain(trainDenseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.setPriorTypes({PriorTypes::normalone, PriorTypes::normalone});

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(matrixSession->getResultItems(), tensorSession->getResultItems());
}

//
//      train: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: normalone normalone
//   aux-data: none none
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_sparse_matrix>    --test <test_sparse_matrix>    --prior normalone normalone --aux-data none none"
   "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior normalone normalone --aux-data none none"
   , TAG_VS_TESTS
)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig = getTestsSmurffConfig();
   matrixSessionConfig.setTrain(trainSparseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.setPriorTypes({PriorTypes::normalone, PriorTypes::normalone});

   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig = getTestsSmurffConfig();
   tensorSessionConfig.setTrain(trainSparseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.setPriorTypes({PriorTypes::normalone, PriorTypes::normalone});

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(matrixSession->getResultItems(), tensorSession->getResultItems());
}

//==========================================================================

//
//      train: 1. dense 2D-tensor (matrix)
//             2. dense matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: macau macau
//  side-info: row_side_info_dense_matrix col_side_info_dense_matrix
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior macau macau --side-info <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   "--train <train_dense_matrix>    --test <test_sparse_matrix>    --prior macau macau --side-info <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   , TAG_VS_TESTS)
{
   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<SideInfoConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseConfig();
   std::shared_ptr<SideInfoConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseConfig();

   Config tensorRunConfig = getTestsSmurffConfig();
   tensorRunConfig.setTrain(trainDenseTensorConfig);
   tensorRunConfig.setTest(testSparseTensorConfig);
   tensorRunConfig.setPriorTypes({PriorTypes::macau, PriorTypes::macau});
   tensorRunConfig.addSideInfoConfig(0, rowSideInfoDenseMatrixConfig);
   tensorRunConfig.addSideInfoConfig(1, colSideInfoDenseMatrixConfig);

   Config matrixRunConfig = getTestsSmurffConfig();
   matrixRunConfig.setTrain(trainDenseMatrixConfig);
   matrixRunConfig.setTest(testSparseMatrixConfig);
   matrixRunConfig.setPriorTypes({PriorTypes::macau, PriorTypes::macau});
   matrixRunConfig.addSideInfoConfig(0, rowSideInfoDenseMatrixConfig);
   matrixRunConfig.addSideInfoConfig(1, colSideInfoDenseMatrixConfig);

   std::shared_ptr<ISession> tensorRunSession = SessionFactory::create_session(tensorRunConfig);
   tensorRunSession->run();

   std::shared_ptr<ISession> matrixRunSession = SessionFactory::create_session(matrixRunConfig);
   matrixRunSession->run();

   double tensorRunRmseAvg = tensorRunSession->getRmseAvg();
   const std::vector<ResultItem> & tensorRunResults = tensorRunSession->getResultItems();

   double matrixRunRmseAvg = matrixRunSession->getRmseAvg();
   const std::vector<ResultItem> & matrixRunResults = matrixRunSession->getResultItems();

   REQUIRE(tensorRunRmseAvg == Approx(matrixRunRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(tensorRunResults, matrixRunResults);
}

//
//      train: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: macau macau
//  side-info: row_side_info_dense_matrix col_side_info_dense_matrix
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior macau macau --side-info <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   "--train <train_sparse_matrix>    --test <test_sparse_matrix>    --prior macau macau --side-info <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   , TAG_VS_TESTS)
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<SideInfoConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseConfig();
   std::shared_ptr<SideInfoConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseConfig();

   Config tensorRunConfig = getTestsSmurffConfig();
   tensorRunConfig.setTrain(trainSparseTensorConfig);
   tensorRunConfig.setTest(testSparseTensorConfig);
   tensorRunConfig.setPriorTypes({PriorTypes::macau, PriorTypes::macau});
   tensorRunConfig.addSideInfoConfig(0, rowSideInfoDenseMatrixConfig);
   tensorRunConfig.addSideInfoConfig(1, colSideInfoDenseMatrixConfig);

   Config matrixRunConfig = getTestsSmurffConfig();
   matrixRunConfig.setTrain(trainSparseMatrixConfig);
   matrixRunConfig.setTest(testSparseMatrixConfig);
   matrixRunConfig.setPriorTypes({PriorTypes::macau, PriorTypes::macau});
   matrixRunConfig.addSideInfoConfig(0, rowSideInfoDenseMatrixConfig);
   matrixRunConfig.addSideInfoConfig(1, colSideInfoDenseMatrixConfig);

   std::shared_ptr<ISession> tensorRunSession = SessionFactory::create_session(tensorRunConfig);
   tensorRunSession->run();

   std::shared_ptr<ISession> matrixRunSession = SessionFactory::create_session(matrixRunConfig);
   matrixRunSession->run();

   double tensorRunRmseAvg = tensorRunSession->getRmseAvg();
   const std::vector<ResultItem> & tensorRunResults = tensorRunSession->getResultItems();

   double matrixRunRmseAvg = matrixRunSession->getRmseAvg();
   const std::vector<ResultItem> & matrixRunResults = matrixRunSession->getResultItems();

   REQUIRE(tensorRunRmseAvg == Approx(matrixRunRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(tensorRunResults, matrixRunResults);
}

//
//      train: 1. dense 2D-tensor (matrix)
//             2. dense matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: macauone macauone
//  side-info: row_side_info_dense_matrix col_side_info_dense_matrix
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior macauone macauone --side-info <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   "--train <train_dense_matrix>    --test <test_sparse_matrix>    --prior macauone macauone --side-info <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   , TAG_VS_TESTS)
{
   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<SideInfoConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseConfig();
   std::shared_ptr<SideInfoConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseConfig();

   Config tensorRunConfig = getTestsSmurffConfig();
   tensorRunConfig.setTrain(trainDenseTensorConfig);
   tensorRunConfig.setTest(testSparseTensorConfig);
   tensorRunConfig.setPriorTypes({PriorTypes::macauone, PriorTypes::macauone});
   tensorRunConfig.addSideInfoConfig(0, rowSideInfoDenseMatrixConfig);
   tensorRunConfig.addSideInfoConfig(1, colSideInfoDenseMatrixConfig);

   Config matrixRunConfig = getTestsSmurffConfig();
   matrixRunConfig.setTrain(trainDenseMatrixConfig);
   matrixRunConfig.setTest(testSparseMatrixConfig);
   matrixRunConfig.setPriorTypes({PriorTypes::macauone, PriorTypes::macauone});
   matrixRunConfig.addSideInfoConfig(0, rowSideInfoDenseMatrixConfig);
   matrixRunConfig.addSideInfoConfig(1, colSideInfoDenseMatrixConfig);

   std::shared_ptr<ISession> tensorRunSession = SessionFactory::create_session(tensorRunConfig);
   tensorRunSession->run();

   std::shared_ptr<ISession> matrixRunSession = SessionFactory::create_session(matrixRunConfig);
   matrixRunSession->run();

   double tensorRunRmseAvg = tensorRunSession->getRmseAvg();
   // const std::vector<ResultItem> & tensorRunResults = tensorRunSession->getResultItems();

   double matrixRunRmseAvg = matrixRunSession->getRmseAvg();
   // const std::vector<ResultItem> & matrixRunResults = matrixRunSession->getResultItems();

   REQUIRE(tensorRunRmseAvg == Approx(matrixRunRmseAvg).epsilon(APPROX_EPSILON));
   // REQUIRE_RESULT_ITEMS(tensorRunResults, matrixRunResults);
}

//
//      train: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: macauone macauone
//  side-info: row_side_info_dense_matrix col_side_info_dense_matrix
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior macauone macauone --side-info <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   "--train <train_sparse_matrix>    --test <test_sparse_matrix>    --prior macauone macauone --side-info <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
   , TAG_VS_TESTS)
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<SideInfoConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseConfig();
   std::shared_ptr<SideInfoConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseConfig();

   Config tensorRunConfig = getTestsSmurffConfig();
   tensorRunConfig.setTrain(trainSparseTensorConfig);
   tensorRunConfig.setTest(testSparseTensorConfig);
   tensorRunConfig.setPriorTypes({PriorTypes::macauone, PriorTypes::macauone});
   tensorRunConfig.addSideInfoConfig(0, rowSideInfoDenseMatrixConfig);
   tensorRunConfig.addSideInfoConfig(1, colSideInfoDenseMatrixConfig);

   Config matrixRunConfig = getTestsSmurffConfig();
   matrixRunConfig.setTrain(trainSparseMatrixConfig);
   matrixRunConfig.setTest(testSparseMatrixConfig);
   matrixRunConfig.setPriorTypes({PriorTypes::macauone, PriorTypes::macauone});
   matrixRunConfig.addSideInfoConfig(0, rowSideInfoDenseMatrixConfig);
   matrixRunConfig.addSideInfoConfig(1, colSideInfoDenseMatrixConfig);

   std::shared_ptr<ISession> tensorRunSession = SessionFactory::create_session(tensorRunConfig);
   tensorRunSession->run();

   std::shared_ptr<ISession> matrixRunSession = SessionFactory::create_session(matrixRunConfig);
   matrixRunSession->run();

   double tensorRunRmseAvg = tensorRunSession->getRmseAvg();
   const std::vector<ResultItem> & tensorRunResults = tensorRunSession->getResultItems();

   double matrixRunRmseAvg = matrixRunSession->getRmseAvg();
   const std::vector<ResultItem> & matrixRunResults = matrixRunSession->getResultItems();

   REQUIRE(tensorRunRmseAvg == Approx(matrixRunRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(tensorRunResults, matrixRunResults);
}

TEST_CASE("PredictSession/BPMF")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config = getTestsSmurffConfig();
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.setPriorTypes({PriorTypes::normal, PriorTypes::normal});
   config.setSaveFreq(1);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   //std::cout << "Prediction from Session RMSE: " << session->getRmseAvg() << std::endl;

   std::string root_fname =  session->getRootFile()->getFullPath();
   auto rf = std::make_shared<RootFile>(root_fname);

   {
       PredictSession s(rf);

       // test predict from TensorConfig
       auto result = s.predict(config.getTest());

       // std::cout << "Prediction from RootFile RMSE: " << result->rmse_avg << std::endl;
       REQUIRE(session->getRmseAvg()  == Approx(result->rmse_avg).epsilon(APPROX_EPSILON));
    }

    {
        PredictSession s(rf, config);
        s.run();
        auto result = s.getResult();

        //std::cout << "Prediction from RootFile+Config RMSE: " << result->rmse_avg << std::endl;
        REQUIRE(session->getRmseAvg()  == Approx(result->rmse_avg).epsilon(APPROX_EPSILON));
    }
}

//=================================================================

//
//      train: dense matrix
//       test: sparse matrix
//     priors: macau normal
//   features: row_side_info_dense_matrix none
//     direct: true
//
TEST_CASE("PredictSession/Features/1"
   , TAG_MATRIX_TESTS)
{
    std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
    std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
    std::shared_ptr<SideInfoConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseConfig();

    Config config = getTestsSmurffConfig();
    config.setTrain(trainDenseMatrixConfig);
    config.setTest(testSparseMatrixConfig);
    config.setPriorTypes({PriorTypes::macau, PriorTypes::normal});
    config.addSideInfoConfig(0, rowSideInfoDenseMatrixConfig);
    config.setSaveFreq(1);

    std::shared_ptr<ISession> session = SessionFactory::create_session(config);
    session->run();

    PredictSession predict_session(session->getRootFile());

    auto sideInfoMatrix = matrix_utils::dense_to_eigen(*rowSideInfoDenseMatrixConfig->getSideInfo());
    auto trainMatrix = smurff::matrix_utils::dense_to_eigen(*trainDenseMatrixConfig);

    #if 0
    std::cout << "sideInfo =\n" << sideInfoMatrix << std::endl;
    std::cout << "train    =\n" << trainMatrix << std::endl;
    #endif

    for(int r = 0; r<sideInfoMatrix.rows(); r++)
    {
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


TEST_CASE("PredictSession/Features/2"
   , TAG_MATRIX_TESTS)
{
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

    std::shared_ptr<MatrixConfig> trainMatrixConfig;
    {
        std::vector<std::uint32_t> trainMatrixConfigRows = {0, 0, 1, 1, 2, 2};
        std::vector<std::uint32_t> trainMatrixConfigCols = {0, 1, 2, 3, 0, 1};
        std::vector<double> trainMatrixConfigVals = {2, 2, 2, 4, -2, -2};
        //std::vector<std::uint32_t> trainMatrixConfigRows = {0, 0, 1, 1, 2, 2, 3, 3};
        //std::vector<std::uint32_t> trainMatrixConfigCols = {0, 1, 2, 3, 0, 1, 2, 3};
        //std::vector<double> trainMatrixConfigVals = {2, 2, 2, 4, -2, -2, -2, -4};
        fixed_ncfg.setPrecision(1.);
        trainMatrixConfig = std::make_shared<MatrixConfig>(4, 4, trainMatrixConfigRows, trainMatrixConfigCols,
            trainMatrixConfigVals, fixed_ncfg, true);
    }

    std::shared_ptr<MatrixConfig> testMatrixConfig;
    {
        std::vector<std::uint32_t> testMatrixConfigRows = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
        std::vector<std::uint32_t> testMatrixConfigCols = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
        std::vector<double> testMatrixConfigVals = {
            2, 2, 1, 2,
            4, 4, 2, 4,
            -2, -2, -1, -2,
            -4, -4, -2, -4};
        testMatrixConfig =
            std::make_shared<MatrixConfig>(4, 4, testMatrixConfigRows, testMatrixConfigCols, testMatrixConfigVals, fixed_ncfg, true);
    }

    std::shared_ptr<SideInfoConfig> rowSideInfoConfig;
    {
        NoiseConfig nc(NoiseTypes::sampled);
        nc.setPrecision(10.0);

        std::vector<std::uint32_t> rowSideInfoSparseMatrixConfigRows = {0, 1, 2, 3};
        std::vector<std::uint32_t> rowSideInfoSparseMatrixConfigCols = {0, 0, 0, 0};
        std::vector<double> rowSideInfoSparseMatrixConfigVals = {2, 4, -2, -4};

        auto mcfg =
            std::make_shared<MatrixConfig>(4, 1, rowSideInfoSparseMatrixConfigRows,
                rowSideInfoSparseMatrixConfigCols, rowSideInfoSparseMatrixConfigVals, nc, true);

        rowSideInfoConfig = std::make_shared<SideInfoConfig>();
        rowSideInfoConfig->setSideInfo(mcfg);
        rowSideInfoConfig->setDirect(true);
    }
    Config config = getTestsSmurffConfig();
    config.setTrain(trainMatrixConfig);
    config.setTest(testMatrixConfig);
    config.setPriorTypes({PriorTypes::macau, PriorTypes::normal});
    config.addSideInfoConfig(0, rowSideInfoConfig);
    config.setSaveFreq(1);

    std::shared_ptr<ISession> session = SessionFactory::create_session(config);
    session->run();

    PredictSession predict_session_in(session->getRootFile());
    auto in_matrix_predictions = predict_session_in.predict(config.getTest())->m_predictions;

    PredictSession predict_session_out(session->getRootFile());
    auto sideInfoMatrix = matrix_utils::sparse_to_eigen(*rowSideInfoConfig->getSideInfo());
    int d = config.getTrain()->getDims()[0];
    for (int r = 0; r < d; r++)
    {
        auto feat = sideInfoMatrix.row(r).transpose();
        auto out_of_matrix_predictions = predict_session_out.predict(0, feat);
        //Vector out_of_matrix_averages = out_of_matrix_predictions->colwise().mean();

#undef DEBUG_OOM_PREDICT
#ifdef DEBUG_OOM_PREDICT
        for (auto p : in_matrix_predictions)
        {
            if (p.coords[0] == r) {
                std::cout << "in: " << p << std::endl;
                std::cout << "  out: " << out_of_matrix_averages.row(p.coords[1]) << std::endl;
            }
        }
#endif
    }
}
