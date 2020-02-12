#include <cstdio>
#include <fstream>

#include "catch.hpp"

#include <SmurffCpp/Types.h>

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Predict/PredictSession.h>
#include <SmurffCpp/Sessions/SessionFactory.h>
#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/RootFile.h>
#include <SmurffCpp/result.h>

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

// Code for printing test results that can then be copy-pasted into tests as
// expected results
void printActualResults(int nr, double actualRmseAvg, const std::vector<smurff::ResultItem> &actualResults) {

  static const char *fname = "TestsSmurff_ExpectedResults.h";
  static bool cleanup = true;

  if (cleanup) {
    std::remove(fname);
    cleanup = false;
  }

  std::ofstream os(fname, std::ofstream::app);

  os << "{ " << nr << ",\n"
     << "  { " << std::fixed << std::setprecision(16) << actualRmseAvg << "," << std::endl
     << "      {\n";

  for (const auto &actualResultItem : actualResults) {
    os << std::setprecision(16);
    os << "         { { " << actualResultItem.coords << " }, " << actualResultItem.val << ", " << std::fixed << actualResultItem.pred_1sample << ", " << actualResultItem.pred_avg << ", " << actualResultItem.var << ", "
       << " }," << std::endl;
  }

  os << "      }\n"
     << "  }\n"
     << "},\n";
}

#define PRINT_ACTUAL_RESULTS(nr)
//#define PRINT_ACTUAL_RESULTS(nr) printActualResults(nr, actualRmseAvg, actualResults);

using namespace smurff;

struct ExpectedResult {
  double rmseAvg;
  std::vector<ResultItem> resultItems;
};
std::map<int, ExpectedResult> expectedResults = {
#include "TestsSmurff_ExpectedResults.h"
};

static NoiseConfig fixed_ncfg(NoiseTypes::fixed);

template <class C> Config genConfig(const C &train, const C &test, std::vector<PriorTypes> priors) {
  Config config;
  config.setBurnin(50);
  config.setNSamples(50);
  config.setVerbose(false);
  config.setRandomSeed(1234);
  config.setNumThreads(1);
  config.setNumLatent(4);
  config.setTrain(std::make_shared<C>(train));
  config.setTest(std::make_shared<C>(test));
  config.setPriorTypes(priors);
  return config;
}

// dense train data (matrix/tensor 2d/tensor 3d)
MatrixConfig trainDenseMatrix(3, 4, {1., 5., 9., 2., 6., 10., 3., 7., 11., 4., 8., 12.}, fixed_ncfg);
TensorConfig trainDenseTensor2d({3, 4}, {1., 5., 9., 2., 6., 10., 3., 7., 11., 4., 8., 12.}, fixed_ncfg);
TensorConfig trainDenseTensor3d({2, 3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.}, fixed_ncfg);

// sparse train data (matrix/tensor 2d)
MatrixConfig trainSparseMatrix(3, 4, {0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}, {1., 2., 3., 4., 9., 10., 11., 12.}, fixed_ncfg, true);
TensorConfig trainSparseTensor2d({3, 4}, {{0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}}, {1., 2., 3., 4., 9., 10., 11., 12.}, fixed_ncfg, true);

// sparse test data (matrix/tensor 2d/tensor 3d)
MatrixConfig testSparseMatrix(3, 4, {0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}, {1., 2., 3., 4., 9., 10., 11., 12.}, fixed_ncfg, true);
TensorConfig testSparseTensor2d({3, 4}, {{0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}}, {1., 2., 3., 4., 9., 10., 11., 12.}, fixed_ncfg, true);
TensorConfig testSparseTensor3d({2, 3, 4}, {{0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}}, {1., 2., 3., 4., 9., 10., 11., 12.}, fixed_ncfg, true);

// aux data

MatrixConfig rowAuxDense(3, 1, {1., 2., 3.}, fixed_ncfg, {0, 1});
MatrixConfig colAuxDense(1, 4, {1., 2., 3., 4.}, fixed_ncfg, {1, 0});

// side info

static NoiseConfig sampled_nc = []() {
  NoiseConfig nc(NoiseTypes::sampled);
  nc.setPrecision(10.0);
  return nc;
}();

static MatrixConfig rowSideDenseMatrix(3, 1, {1., 2., 3.}, sampled_nc);
static MatrixConfig colSideDenseMatrix(4, 1, {1., 2., 3., 4.}, sampled_nc);
static MatrixConfig rowSideSparseMatrix(3, 1, {0, 1, 2}, {0, 0, 0}, {1., 2., 3.}, sampled_nc, false);
static MatrixConfig colSideSparseMatrix(4, 1, {0, 1, 2, 3}, {0, 0, 0, 0}, {1., 2., 3., 4.}, sampled_nc, false);
static MatrixConfig rowSideDenseMatrix3d(2, 3, {1., 2., 3., 4., 5., 6.}, sampled_nc);

std::shared_ptr<SideInfoConfig> toSide(const MatrixConfig &mcfg, bool direct = true, double tol = 1e-6) {
  std::shared_ptr<SideInfoConfig> picfg = std::make_shared<SideInfoConfig>();
  picfg->setSideInfo(std::make_shared<MatrixConfig>(mcfg));
  picfg->setDirect(direct);
  picfg->setTol(tol);

  return picfg;
}

// result comparison

void REQUIRE_RESULT_ITEMS(const std::vector<ResultItem> &actualResultItems, const std::vector<ResultItem> &expectedResultItems) {
  REQUIRE(actualResultItems.size() == expectedResultItems.size());
  double single_item_epsilon = APPROX_EPSILON * 10;
  for (std::vector<ResultItem>::size_type i = 0; i < actualResultItems.size(); i++) {
    const ResultItem &actualResultItem = actualResultItems[i];
    const ResultItem &expectedResultItem = expectedResultItems[i];
    REQUIRE(actualResultItem.coords == expectedResultItem.coords);
    REQUIRE(actualResultItem.val == expectedResultItem.val);
    REQUIRE(actualResultItem.pred_1sample == Approx(expectedResultItem.pred_1sample).epsilon(single_item_epsilon));
    REQUIRE(actualResultItem.pred_avg == Approx(expectedResultItem.pred_avg).epsilon(single_item_epsilon));
    REQUIRE(actualResultItem.var == Approx(expectedResultItem.var).epsilon(single_item_epsilon));
  }
}


struct SmurffTest {
  Config config;

  SmurffTest(const MatrixConfig &train, const MatrixConfig &test, std::vector<PriorTypes> priors) : config(genConfig(train, test, priors)) {}

  SmurffTest(const TensorConfig &train, const TensorConfig &test, std::vector<PriorTypes> priors) : config(genConfig(train, test, priors)) {}

  SmurffTest &addSideInfoConfig(int m, const MatrixConfig &c,  bool direct = true, double tol = 1e-6)
  {
      std::shared_ptr<SideInfoConfig> picfg = std::make_shared<SideInfoConfig>();
      picfg->setSideInfo(std::make_shared<MatrixConfig>(c));
      picfg->setDirect(direct);
      picfg->setTol(tol);
      config.addSideInfoConfig(m, picfg);
      return *this;
  }

  SmurffTest &addAuxData(const TensorConfig &c) {
    config.addAuxData(std::make_shared<TensorConfig>(c));
    return *this;
  }

  SmurffTest &addAuxData(std::shared_ptr<TensorConfig> c) {
    config.addAuxData(c);
    return *this;
  }

  void runAndCheck(int nr) {
      std::shared_ptr<ISession> session = SessionFactory::create_session(config);
      session->run();

      double actualRmseAvg = session->getRmseAvg();
      const std::vector<ResultItem> &actualResults = session->getResultItems();

      PRINT_ACTUAL_RESULTS(nr)
          double &expectedRmseAvg = expectedResults[nr].rmseAvg;
      auto &expectedResultItems = expectedResults[nr].resultItems;

      REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
      REQUIRE_RESULT_ITEMS(actualResults, expectedResultItems);
  }

};

void compareSessions(Config &matrixSessionConfig, Config &tensorSessionConfig) {
  std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
  std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
  matrixSession->run();
  tensorSession->run();

  REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
  REQUIRE_RESULT_ITEMS(matrixSession->getResultItems(), tensorSession->getResultItems());
}

///===========================================================================

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior "
          "normal normal --aux-data none none",
          TAG_MATRIX_TESTS) {

  // SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal}).runAndCheck(359);
  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal}).runAndCheck(359);
}

TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior "
          "normal normal --aux-data none none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal}).runAndCheck(411);
}

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior "
          "normal normal --aux-data <dense_matrix> <dense_matrix>",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal}).addAuxData(rowAuxDense).addAuxData(colAuxDense).runAndCheck(467);
}

TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior "
          "normal normal --aux-data <dense_matrix> <dense_matrix>",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal}).addAuxData(rowAuxDense).addAuxData(colAuxDense).runAndCheck(523);
}

//=================================================================

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior "
          "spikeandslab spikeandslab --aux-data none none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::spikeandslab}).runAndCheck(577);
}

TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior "
          "spikeandslab spikeandslab --aux-data none none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::spikeandslab}).runAndCheck(629);
}

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior "
          "spikeandslab spikeandslab --aux-data <dense_matrix> <dense_matrix>",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::spikeandslab}).addAuxData(rowAuxDense).addAuxData(colAuxDense).runAndCheck(685);
}

TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior "
          "spikeandslab spikeandslab --aux-data <dense_matrix> <dense_matrix>",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::spikeandslab}).addAuxData(rowAuxDense).addAuxData(colAuxDense).runAndCheck(741);
}

//=================================================================

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior "
          "normalone normalone --aux-data none none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normalone, PriorTypes::normalone}).runAndCheck(795);
}

TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior "
          "normalone normalone --aux-data none none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::normalone, PriorTypes::normalone}).runAndCheck(847);
}

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior "
          "normalone normalone --aux-data <dense_matrix> <dense_matrix>",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normalone, PriorTypes::normalone}).addAuxData(rowAuxDense).addAuxData(colAuxDense).runAndCheck(903);
}

TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior "
          "normalone normalone --aux-data <dense_matrix> <dense_matrix>",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::normalone, PriorTypes::normalone}).addAuxData(rowAuxDense).addAuxData(colAuxDense).runAndCheck(959);
}

//=================================================================

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior "
          "macau macau --aux-data <row_side_info_dense_matrix> "
          "<col_side_info_dense_matrix> --direct",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::macau}).addSideInfoConfig(0, rowSideDenseMatrix).addSideInfoConfig(1, colSideDenseMatrix).runAndCheck(1018);
}

TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior "
          "macau macau --aux-data <row_side_info_dense_matrix> "
          "<col_side_info_dense_matrix> --direct",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::macau}).addSideInfoConfig(0, rowSideDenseMatrix).addSideInfoConfig(1, colSideDenseMatrix).runAndCheck(1075);
}

//=================================================================

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior "
          "macauone macauone --aux-data <row_side_info_sparse_matrix> "
          "<col_side_info_sparse_matrix> --direct",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::macauone, PriorTypes::macauone}).addSideInfoConfig(0, rowSideSparseMatrix).addSideInfoConfig(1, colSideSparseMatrix).runAndCheck(1135);
}

TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior "
          "macauone macauone --aux-data <row_side_info_sparse_matrix> "
          "<col_side_info_sparse_matrix> --direct",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::macauone, PriorTypes::macauone}).addSideInfoConfig(0, rowSideSparseMatrix).addSideInfoConfig(1, colSideSparseMatrix).runAndCheck(1193);
}

//=================================================================

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior "
          "macau normal --aux-data <row_side_info_dense_matrix> none --direct",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::normal}).addSideInfoConfig(0, rowSideDenseMatrix).runAndCheck(1250);
}

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior "
          "normal macau --aux-data none <col_side_info_dense_matrix> --direct",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::macau}).addSideInfoConfig(1, colSideDenseMatrix).runAndCheck(1305);
}

// test throw - macau prior should have side info

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior "
          "macau normal --aux-data none none --direct",
          TAG_MATRIX_TESTS) {

  Config config = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::normal}).addSideInfoConfig(1, toSide(rowSideDenseMatrix));

  REQUIRE_THROWS(SessionFactory::create_session(config));
}

// test throw - wrong dimentions of side info

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior "
          "macau normal --aux-data <col_side_info_dense_matrix> none --direct",
          TAG_MATRIX_TESTS) {

  Config config = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::normal}).addSideInfoConfig(1, toSide(colSideDenseMatrix));

  REQUIRE_THROWS(SessionFactory::create_session(config));
}

//=================================================================

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior "
          "normal spikeandslab --aux-data none none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::spikeandslab}).runAndCheck(1466);
}

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior "
          "spikeandslab normal --aux-data none none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::normal}).runAndCheck(1518);
}

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior "
          "normal spikeandslab --aux-data none <dense_matrix>",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::normal}).addAuxData(colAuxDense).runAndCheck(1572);
}

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior "
          "spikeandslab normal --aux-data <dense_matrix> none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::normal}).addAuxData(rowAuxDense).runAndCheck(1626);
}

//=================================================================

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior macau "
          "spikeandslab --aux-data <row_side_info_dense_matrix> none --direct",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::spikeandslab}).addSideInfoConfig(0, rowSideDenseMatrix).runAndCheck(1683);
}

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior "
          "spikeandslab macau --aux-data none <col_side_info_dense_matrix> --direct",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::macau}).addSideInfoConfig(1, colSideDenseMatrix).runAndCheck(1738);
}

//=================================================================

TEST_CASE("--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normal normal --aux-data none none",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {

  SmurffTest(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::normal}).runAndCheck(1792);
}

TEST_CASE("--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normal normal --aux-data none none",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::normal}).runAndCheck(1844);
}

//=================================================================
TEST_CASE("--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior spikeandslab spikeandslab --aux-data none none",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::spikeandslab, PriorTypes::spikeandslab}).runAndCheck(1898);
}

TEST_CASE("--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior spikeandslab spikeandslab --aux-data none none",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::spikeandslab, PriorTypes::spikeandslab}).runAndCheck(1950);
}

//=================================================================

TEST_CASE("--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normalone normalone --aux-data none none",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::normalone, PriorTypes::normalone}).runAndCheck(2004);
}

TEST_CASE("--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normalone normalone --aux-data none none",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::normalone, PriorTypes::normalone}).runAndCheck(2056);
}

//=================================================================

TEST_CASE("--train <train_dense_3d_tensor> --test <test_sparse_3d_tensor> "
          "--prior normal normal --aux-data none none",
          TAG_THREE_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainDenseTensor3d, testSparseTensor3d, {PriorTypes::normal, PriorTypes::normal, PriorTypes::normal}).runAndCheck(2110);
}

//=================================================================

TEST_CASE("--train <train_dense_3d_tensor> --test <test_sparse_3d_tensor> "
          "--prior spikeandslab spikeandslab --aux-data none none",
          TAG_THREE_DIMENTIONAL_TENSOR_TESTS) {

  SmurffTest(trainDenseTensor3d, testSparseTensor3d, {PriorTypes::spikeandslab, PriorTypes::spikeandslab, PriorTypes::spikeandslab}).runAndCheck(2164);
}

//=================================================================

// not sure if this test produces correct results

TEST_CASE("--train <train_dense_3d_tensor> --test <test_sparse_3d_tensor> "
          "--prior macau normal --side-info row_dense_side_info none",
          TAG_THREE_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainDenseTensor3d, testSparseTensor3d, {PriorTypes::macau, PriorTypes::normal, PriorTypes::normal}).addSideInfoConfig(0, rowSideDenseMatrix3d).runAndCheck(2222);
}

//=================================================================

// not sure if this test produces correct results

TEST_CASE("--train <train_dense_3d_tensor> --test <test_sparse_3d_tensor> "
          "--prior macauone normal --side-info row_dense_side_info none",
          TAG_THREE_DIMENTIONAL_TENSOR_TESTS "[!mayfail]") {
  SmurffTest(trainDenseTensor3d, testSparseTensor3d, {PriorTypes::macauone, PriorTypes::normal, PriorTypes::normal}).addSideInfoConfig(0, rowSideDenseMatrix3d).runAndCheck(2280);
}

//=================================================================

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    "
          "--prior normal normal --aux-data none none"
          "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normal normal --aux-data none none",
          TAG_VS_TESTS) {
  Config matrixSessionConfig = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal});
  Config tensorSessionConfig = genConfig(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::normal});
  compareSessions(matrixSessionConfig, tensorSessionConfig);
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior normal normal --aux-data none none"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normal normal --aux-data none none",
          TAG_VS_TESTS) {
  Config matrixSessionConfig = genConfig(trainSparseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal});
  Config tensorSessionConfig = genConfig(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::normal});
  compareSessions(matrixSessionConfig, tensorSessionConfig);
}

//             2. dense matrix
//             2. sparse matrix
TEST_CASE("matrix vs 2D-tensor"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    "
          "--prior normal spikeandslab --aux-data none none"
          "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normal spikeandslab --aux-data none none",
          TAG_VS_TESTS) {
  Config matrixSessionConfig = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::spikeandslab});
  Config tensorSessionConfig = genConfig(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::spikeandslab});
  compareSessions(matrixSessionConfig, tensorSessionConfig);
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior normal spikeandslab --aux-data none none"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normal spikeandslab --aux-data none none",
          TAG_VS_TESTS) {
  Config matrixSessionConfig = genConfig(trainSparseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::spikeandslab});
  Config tensorSessionConfig = genConfig(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::spikeandslab});
  compareSessions(matrixSessionConfig, tensorSessionConfig);
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    "
          "--prior spikeandslab normal --aux-data none none"
          "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior spikeandslab normal --aux-data none none",
          TAG_VS_TESTS) {
  Config matrixSessionConfig = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::normal});
  Config tensorSessionConfig = genConfig(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::spikeandslab, PriorTypes::normal});
  compareSessions(matrixSessionConfig, tensorSessionConfig);
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior spikeandslab normal --aux-data none none"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior spikeandslab normal --aux-data none none",
          TAG_VS_TESTS) {
  Config matrixSessionConfig = genConfig(trainSparseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::normal});
  Config tensorSessionConfig = genConfig(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::spikeandslab, PriorTypes::normal});
  compareSessions(matrixSessionConfig, tensorSessionConfig);
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    "
          "--prior spikeandslab spikeandslab --aux-data none none"
          "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior spikeandslab spikeandslab --aux-data none none",
          TAG_VS_TESTS) {
  Config matrixSessionConfig = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::spikeandslab});
  Config tensorSessionConfig = genConfig(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::spikeandslab, PriorTypes::spikeandslab});
  compareSessions(matrixSessionConfig, tensorSessionConfig);
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior spikeandslab spikeandslab --aux-data none none"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior spikeandslab spikeandslab --aux-data none none",
          TAG_VS_TESTS) {
  Config matrixSessionConfig = genConfig(trainSparseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::spikeandslab});
  Config tensorSessionConfig = genConfig(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::spikeandslab, PriorTypes::spikeandslab});
  compareSessions(matrixSessionConfig, tensorSessionConfig);
}

//==========================================================================

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    "
          "--prior normal normalone --aux-data none none"
          "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normal normalone --aux-data none none",
          TAG_VS_TESTS) {
  Config matrixSessionConfig = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normalone});
  Config tensorSessionConfig = genConfig(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::normalone});
  compareSessions(matrixSessionConfig, tensorSessionConfig);
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior normal normalone --aux-data none none"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normal normalone --aux-data none none",
          TAG_VS_TESTS) {
  Config matrixSessionConfig = genConfig(trainSparseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normalone});
  Config tensorSessionConfig = genConfig(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::normalone});
  compareSessions(matrixSessionConfig, tensorSessionConfig);
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    "
          "--prior normalone normal --aux-data none none"
          "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normalone normal --aux-data none none",
          TAG_VS_TESTS) {
  Config matrixSessionConfig = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::normalone, PriorTypes::normal});
  Config tensorSessionConfig = genConfig(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::normalone, PriorTypes::normal});
  compareSessions(matrixSessionConfig, tensorSessionConfig);
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior normalone normal --aux-data none none"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normalone normal --aux-data none none",
          TAG_VS_TESTS) {
  Config matrixSessionConfig = genConfig(trainSparseMatrix, testSparseMatrix, {PriorTypes::normalone, PriorTypes::normal});
  Config tensorSessionConfig = genConfig(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::normalone, PriorTypes::normal});
  compareSessions(matrixSessionConfig, tensorSessionConfig);
}

//             2. dense matrix
//             2. sparse matrix
TEST_CASE("matrix vs 2D-tensor"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    "
          "--prior normalone normalone --aux-data none none"
          "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normalone normalone --aux-data none none",
          TAG_VS_TESTS) {
  Config matrixSessionConfig = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::normalone, PriorTypes::normalone});
  Config tensorSessionConfig = genConfig(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::normalone, PriorTypes::normalone});
  compareSessions(matrixSessionConfig, tensorSessionConfig);
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior normalone normalone --aux-data none none"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normalone normalone --aux-data none none",
          TAG_VS_TESTS) {
  Config matrixSessionConfig = genConfig(trainSparseMatrix, testSparseMatrix, {PriorTypes::normalone, PriorTypes::normalone});
  Config tensorSessionConfig = genConfig(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::normalone, PriorTypes::normalone});
  compareSessions(matrixSessionConfig, tensorSessionConfig);
}

//==========================================================================

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior macau macau --side-info <row_side_info_dense_matrix> "
          "<col_side_info_dense_matrix> --direct"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    "
          "--prior macau macau --side-info <row_side_info_dense_matrix> "
          "<col_side_info_dense_matrix> --direct",
          TAG_VS_TESTS) {

  Config tensorRunConfig = genConfig(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::macau, PriorTypes::macau}).addSideInfoConfig(0, toSide(rowSideDenseMatrix)).addSideInfoConfig(1, toSide(colSideDenseMatrix));
  Config matrixRunConfig = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::macau}).addSideInfoConfig(0, toSide(rowSideDenseMatrix)).addSideInfoConfig(1, toSide(colSideDenseMatrix));
  compareSessions(tensorRunConfig, matrixRunConfig);
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior macau macau --side-info <row_side_info_dense_matrix> "
          "<col_side_info_dense_matrix> --direct"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior macau macau --side-info <row_side_info_dense_matrix> "
          "<col_side_info_dense_matrix> --direct",
          TAG_VS_TESTS) {

  Config tensorRunConfig = genConfig(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::macau, PriorTypes::macau}).addSideInfoConfig(0, toSide(rowSideDenseMatrix)).addSideInfoConfig(1, toSide(colSideDenseMatrix));
  Config matrixRunConfig = genConfig(trainSparseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::macau}).addSideInfoConfig(0, toSide(rowSideDenseMatrix)).addSideInfoConfig(1, toSide(colSideDenseMatrix));
  compareSessions(tensorRunConfig, matrixRunConfig);
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior macauone macauone --side-info <row_side_info_dense_matrix> "
          "<col_side_info_dense_matrix> --direct"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    "
          "--prior macauone macauone --side-info <row_side_info_dense_matrix> "
          "<col_side_info_dense_matrix> --direct",
          TAG_VS_TESTS) {

  Config tensorRunConfig = genConfig(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::macauone, PriorTypes::macauone}).addSideInfoConfig(0, toSide(rowSideDenseMatrix)).addSideInfoConfig(1, toSide(colSideDenseMatrix));
  Config matrixRunConfig = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::macauone, PriorTypes::macauone}).addSideInfoConfig(0, toSide(rowSideDenseMatrix)).addSideInfoConfig(1, toSide(colSideDenseMatrix));
  compareSessions(tensorRunConfig, matrixRunConfig);
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior macauone macauone --side-info <row_side_info_dense_matrix> "
          "<col_side_info_dense_matrix> --direct"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior macauone macauone --side-info <row_side_info_dense_matrix> "
          "<col_side_info_dense_matrix> --direct",
          TAG_VS_TESTS) {

  Config tensorRunConfig = genConfig(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::macauone, PriorTypes::macauone}).addSideInfoConfig(0, toSide(rowSideDenseMatrix)).addSideInfoConfig(1, toSide(colSideDenseMatrix));
  Config matrixRunConfig = genConfig(trainSparseMatrix, testSparseMatrix, {PriorTypes::macauone, PriorTypes::macauone}).addSideInfoConfig(0, toSide(rowSideDenseMatrix)).addSideInfoConfig(1, toSide(colSideDenseMatrix));
  compareSessions(tensorRunConfig, matrixRunConfig);
}

TEST_CASE("PredictSession/BPMF") {

  Config config = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal});
  config.setSaveFreq(1);

  std::shared_ptr<ISession> session = SessionFactory::create_session(config);
  session->run();

  // std::cout << "Prediction from Session RMSE: " << session->getRmseAvg() <<
  // std::endl;

  std::string root_fname = session->getRootFile()->getFullPath();
  auto rf = std::make_shared<RootFile>(root_fname);

  {
    PredictSession s(rf);

    // test predict from TensorConfig
    auto result = s.predict(config.getTest());

    // std::cout << "Prediction from RootFile RMSE: " << result->rmse_avg <<
    // std::endl;
    REQUIRE(session->getRmseAvg() == Approx(result->rmse_avg).epsilon(APPROX_EPSILON));
  }

  {
    PredictSession s(rf, config);
    s.run();
    auto result = s.getResult();

    // std::cout << "Prediction from RootFile+Config RMSE: " << result->rmse_avg
    // << std::endl;
    REQUIRE(session->getRmseAvg() == Approx(result->rmse_avg).epsilon(APPROX_EPSILON));
  }
}

//=================================================================

TEST_CASE("PredictSession/Features/1", TAG_MATRIX_TESTS) {
  std::shared_ptr<SideInfoConfig> rowSideInfoDenseMatrixConfig = toSide(rowSideDenseMatrix);

  Config config = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::normal}).addSideInfoConfig(0, rowSideInfoDenseMatrixConfig);
  config.setSaveFreq(1);

  std::shared_ptr<ISession> session = SessionFactory::create_session(config);
  session->run();

  PredictSession predict_session(session->getRootFile());

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

  MatrixConfig trainMatrixConfig;
  {
    std::vector<std::uint32_t> trainMatrixConfigRows = {0, 0, 1, 1, 2, 2};
    std::vector<std::uint32_t> trainMatrixConfigCols = {0, 1, 2, 3, 0, 1};
    std::vector<double> trainMatrixConfigVals = {2, 2, 2, 4, -2, -2};
    // std::vector<std::uint32_t> trainMatrixConfigRows = {0, 0, 1, 1, 2, 2, 3,
    // 3}; std::vector<std::uint32_t> trainMatrixConfigCols = {0, 1, 2, 3, 0, 1,
    // 2, 3}; std::vector<double> trainMatrixConfigVals = {2, 2, 2, 4, -2, -2,
    // -2, -4};
    fixed_ncfg.setPrecision(1.);
    trainMatrixConfig = MatrixConfig(4, 4, trainMatrixConfigRows, trainMatrixConfigCols, trainMatrixConfigVals, fixed_ncfg, true);
  }

  MatrixConfig testMatrixConfig;
  {
    std::vector<std::uint32_t> testMatrixConfigRows = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    std::vector<std::uint32_t> testMatrixConfigCols = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> testMatrixConfigVals = {2, 2, 1, 2, 4, 4, 2, 4, -2, -2, -1, -2, -4, -4, -2, -4};
    testMatrixConfig = MatrixConfig(4, 4, testMatrixConfigRows, testMatrixConfigCols, testMatrixConfigVals, fixed_ncfg, true);
  }

  std::shared_ptr<SideInfoConfig> rowSideInfoConfig;
  {
    NoiseConfig nc(NoiseTypes::sampled);
    nc.setPrecision(10.0);

    std::vector<std::uint32_t> rowSideInfoSparseMatrixConfigRows = {0, 1, 2, 3};
    std::vector<std::uint32_t> rowSideInfoSparseMatrixConfigCols = {0, 0, 0, 0};
    std::vector<double> rowSideInfoSparseMatrixConfigVals = {2, 4, -2, -4};

    auto mcfg = std::make_shared<MatrixConfig>(4, 1, rowSideInfoSparseMatrixConfigRows, rowSideInfoSparseMatrixConfigCols, rowSideInfoSparseMatrixConfigVals, nc, true);

    rowSideInfoConfig = std::make_shared<SideInfoConfig>();
    rowSideInfoConfig->setSideInfo(mcfg);
    rowSideInfoConfig->setDirect(true);
  }
  Config config = genConfig(trainMatrixConfig, testMatrixConfig, {PriorTypes::macau, PriorTypes::normal}).addSideInfoConfig(0, rowSideInfoConfig);
  config.setSaveFreq(1);

  std::shared_ptr<ISession> session = SessionFactory::create_session(config);
  session->run();

  PredictSession predict_session_in(session->getRootFile());
  auto in_matrix_predictions = predict_session_in.predict(config.getTest())->m_predictions;

  PredictSession predict_session_out(session->getRootFile());
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
