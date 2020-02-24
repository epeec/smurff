#include <algorithm>
#include <cstdio>
#include <fstream>

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
#define TAG_TWO_DIMENTIONAL_TENSOR_TESTS "[tensor2d][random]"
#define TAG_THREE_DIMENTIONAL_TENSOR_TESTS "[tensor3d][random]"
#else
#define TAG_MATRIX_TESTS "[matrix][random][!mayfail]"
#define TAG_TWO_DIMENTIONAL_TENSOR_TESTS "[tensor2d][random][!mayfail]"
#define TAG_THREE_DIMENTIONAL_TENSOR_TESTS "[tensor3d][random][!mayfail]"
#endif

namespace smurff {
namespace test {

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

  auto sortedResults = actualResults;
  std::sort(sortedResults.begin(), sortedResults.end());

  for (const auto &actualResultItem : actualResults) {
    os << std::setprecision(16);
    os << "         { { " << actualResultItem.coords << " }, " << actualResultItem.val << ", " << std::fixed
       << actualResultItem.pred_1sample << ", " << actualResultItem.pred_avg << ", " << actualResultItem.var << ", "
       << " }," << std::endl;
  }

  os << "      }\n"
     << "  }\n"
     << "},\n";
}

#define PRINT_ACTUAL_RESULTS(nr)
//#define PRINT_ACTUAL_RESULTS(nr) printActualResults(nr, actualRmseAvg, actualResults);

struct ExpectedResult {
  double rmseAvg;
  std::vector<ResultItem> resultItems;
};
std::map<int, ExpectedResult> expectedResults = {
#include "TestsSmurff_ExpectedResults.h"
};

SideInfoConfig makeSideInfoConfig(const MatrixConfig &mcfg, bool direct, double tol) {
  SideInfoConfig picfg;
  if (mcfg.isDense())
    picfg.setData(matrix_utils::dense_to_eigen(mcfg));
  else
    picfg.setData(matrix_utils::sparse_to_eigen(mcfg));
  picfg.setNoiseConfig(mcfg.getNoiseConfig());
  picfg.setDirect(direct);
  picfg.setTol(tol);

  return picfg;
}

// result comparison

void REQUIRE_RESULT_ITEMS(const std::vector<ResultItem> &actualResultItems,
                          const std::vector<ResultItem> &expectedResultItems) {
  REQUIRE(actualResultItems.size() == expectedResultItems.size());
  double single_item_epsilon = APPROX_EPSILON * 10;

  auto sortedActualResultItems = actualResultItems;
  std::sort(sortedActualResultItems.begin(), sortedActualResultItems.end());
  auto sortedExpectedResultItems = expectedResultItems;
  std::sort(sortedExpectedResultItems.begin(), sortedExpectedResultItems.end());

  for (std::vector<ResultItem>::size_type i = 0; i < sortedActualResultItems.size(); i++) {
    const ResultItem &actualResultItem = sortedActualResultItems[i];
    const ResultItem &expectedResultItem = sortedExpectedResultItems[i];
    REQUIRE(actualResultItem.coords == expectedResultItem.coords);
    REQUIRE(actualResultItem.val == expectedResultItem.val);
    REQUIRE(actualResultItem.pred_1sample == Approx(expectedResultItem.pred_1sample).epsilon(single_item_epsilon));
    REQUIRE(actualResultItem.pred_avg == Approx(expectedResultItem.pred_avg).epsilon(single_item_epsilon));
    REQUIRE(actualResultItem.var == Approx(expectedResultItem.var).epsilon(single_item_epsilon));
  }
}

struct SmurffTest {
  Config config;
  SmurffTest(const DenseTensor &train, const MatrixConfig &test, std::vector<PriorTypes> priors)
      : config(genConfig(DataConfig(matrix_utils::dense_to_eigen(train), fixed_ncfg), test, priors)) {}

  SmurffTest(const MatrixConfig &train, const MatrixConfig &test, std::vector<PriorTypes> priors)
      : config(genConfig(train, test, priors)) {}

  SmurffTest(const TensorConfig &train, const TensorConfig &test, std::vector<PriorTypes> priors)
      : config(genConfig(train, test, priors)) {}

  SmurffTest &addSideInfoConfig(int m, const MatrixConfig &c, bool direct = true, double tol = 1e-6) {
    config.addSideInfoConfig(m, makeSideInfoConfig(c, direct, tol));
    return *this;
  }

  SmurffTest &addAuxData(const TensorConfig &c) {
    config.addAuxData(std::make_shared<TensorConfig>(c));
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

///===========================================================================
TEST_CASE("train_dense_dataconfig__test_sparse_matrix__"
          "normal normal",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDense, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal}).runAndCheck(359);
}

TEST_CASE("train_dense_matrix__test_sparse_matrix__"
          "normal normal",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal}).runAndCheck(359);
}

TEST_CASE("train_sparse_matrix__test_sparse_matrix__"
          "normal normal",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal}).runAndCheck(411);
}

TEST_CASE("train_dense_matrix__test_sparse_matrix__"
          "normal normal__dense_matrix_dense_matrix",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal})
      .addAuxData(rowAuxDense)
      .addAuxData(colAuxDense)
      .runAndCheck(467);
}

TEST_CASE("train_sparse_matrix__test_sparse_matrix__"
          "normal normal__dense_matrix_dense_matrix",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal})
      .addAuxData(rowAuxDense)
      .addAuxData(colAuxDense)
      .runAndCheck(523);
}

//=================================================================

TEST_CASE("train_dense_matrix__test_sparse_matrix__"
          "spikeandslab spikeandslab",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::spikeandslab}).runAndCheck(577);
}

TEST_CASE("train_sparse_matrix__test_sparse_matrix__"
          "spikeandslab spikeandslab",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::spikeandslab})
      .runAndCheck(629);
}

TEST_CASE("train_dense_matrix__test_sparse_matrix__"
          "spikeandslab spikeandslab__dense_matrix_dense_matrix",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::spikeandslab})
      .addAuxData(rowAuxDense)
      .addAuxData(colAuxDense)
      .runAndCheck(685);
}

TEST_CASE("train_sparse_matrix__test_sparse_matrix__"
          "spikeandslab spikeandslab__dense_matrix_dense_matrix",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::spikeandslab})
      .addAuxData(rowAuxDense)
      .addAuxData(colAuxDense)
      .runAndCheck(741);
}

//=================================================================

TEST_CASE("train_dense_matrix__test_sparse_matrix__"
          "normalone normalone",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normalone, PriorTypes::normalone}).runAndCheck(795);
}

TEST_CASE("train_sparse_matrix__test_sparse_matrix__"
          "normalone normalone",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::normalone, PriorTypes::normalone}).runAndCheck(847);
}

TEST_CASE("train_dense_matrix__test_sparse_matrix__"
          "normalone normalone__dense_matrix_dense_matrix",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normalone, PriorTypes::normalone})
      .addAuxData(rowAuxDense)
      .addAuxData(colAuxDense)
      .runAndCheck(903);
}

TEST_CASE("train_sparse_matrix__test_sparse_matrix__"
          "normalone normalone__dense_matrix_dense_matrix",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::normalone, PriorTypes::normalone})
      .addAuxData(rowAuxDense)
      .addAuxData(colAuxDense)
      .runAndCheck(959);
}

//=================================================================

TEST_CASE("train_dense_matrix__test_sparse_matrix__"
          "macau macau__row_side_info_dense_matrix_col_side_info_dense_matrix__direct",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::macau})
      .addSideInfoConfig(0, rowSideDenseMatrix)
      .addSideInfoConfig(1, colSideDenseMatrix)
      .runAndCheck(1018);
}

TEST_CASE("train_sparse_matrix__test_sparse_matrix__"
          "macau macau__row_side_info_dense_matrix_col_side_info_dense_matrix__direct",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::macau})
      .addSideInfoConfig(0, rowSideDenseMatrix)
      .addSideInfoConfig(1, colSideDenseMatrix)
      .runAndCheck(1075);
}

//=================================================================

TEST_CASE("train_dense_matrix__test_sparse_matrix__"
          "macauone macauone__row_side_info_sparse_matrix_col_side_info_sparse_matrix__direct",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::macauone, PriorTypes::macauone})
      .addSideInfoConfig(0, rowSideSparseMatrix)
      .addSideInfoConfig(1, colSideSparseMatrix)
      .runAndCheck(1135);
}

TEST_CASE("train_sparse_matrix__test_sparse_matrix__"
          "macauone macauone__row_side_info_sparse_matrix_col_side_info_sparse_matrix__direct",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainSparseMatrix, testSparseMatrix, {PriorTypes::macauone, PriorTypes::macauone})
      .addSideInfoConfig(0, rowSideSparseMatrix)
      .addSideInfoConfig(1, colSideSparseMatrix)
      .runAndCheck(1193);
}

//=================================================================

TEST_CASE("train_dense_matrix__test_sparse_matrix__"
          "macau normal__row_side_info_dense_matrix_none__direct",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::normal})
      .addSideInfoConfig(0, rowSideDenseMatrix)
      .runAndCheck(1250);
}

TEST_CASE("train_dense_matrix__test_sparse_matrix__"
          "normal macau__none_col_side_info_dense_matrix__direct",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::macau})
      .addSideInfoConfig(1, colSideDenseMatrix)
      .runAndCheck(1305);
}

// test throw - macau prior should have side info

TEST_CASE("train_dense_matrix__test_sparse_matrix__"
          "macau normal__direct",
          TAG_MATRIX_TESTS) {

  Config config = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::normal})
                      .addSideInfoConfig(1, makeSideInfoConfig(rowSideDenseMatrix));

  REQUIRE_THROWS(SessionFactory::create_session(config));
}

// test throw - wrong dimentions of side info

TEST_CASE("train_dense_matrix__test_sparse_matrix__"
          "macau normal__col_side_info_dense_matrix_none__direct",
          TAG_MATRIX_TESTS) {

  Config config = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::normal})
                      .addSideInfoConfig(1, makeSideInfoConfig(colSideDenseMatrix));

  REQUIRE_THROWS(SessionFactory::create_session(config));
}

//=================================================================

TEST_CASE("train_dense_matrix__test_sparse_matrix__"
          "normal spikeandslab",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::spikeandslab}).runAndCheck(1466);
}

TEST_CASE("train_dense_matrix__test_sparse_matrix__"
          "spikeandslab normal",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::normal}).runAndCheck(1518);
}

TEST_CASE("train_dense_matrix__test_sparse_matrix__"
          "normal spikeandslab__none_dense_matrix",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::normal})
      .addAuxData(colAuxDense)
      .runAndCheck(1572);
}

TEST_CASE("train_dense_matrix__test_sparse_matrix__"
          "spikeandslab normal__dense_matrix_none",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::normal})
      .addAuxData(rowAuxDense)
      .runAndCheck(1626);
}

//=================================================================

TEST_CASE("train_dense_matrix__test_sparse_matrix_macau_"
          "spikeandslab__row_side_info_dense_matrix_none__direct",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::spikeandslab})
      .addSideInfoConfig(0, rowSideDenseMatrix)
      .runAndCheck(1683);
}

TEST_CASE("train_dense_matrix__test_sparse_matrix__"
          "spikeandslab macau__none_col_side_info_dense_matrix__direct",
          TAG_MATRIX_TESTS) {

  SmurffTest(trainDenseMatrix, testSparseMatrix, {PriorTypes::spikeandslab, PriorTypes::macau})
      .addSideInfoConfig(1, colSideDenseMatrix)
      .runAndCheck(1738);
}

//=================================================================

TEST_CASE("train_dense_2d_tensor__test_sparse_2d_tensor_"
          "normal_normal",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {

  SmurffTest(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::normal}).runAndCheck(1792);
}

TEST_CASE("train_sparse_2d_tensor__test_sparse_2d_tensor_"
          "normal_normal",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::normal}).runAndCheck(1844);
}

//=================================================================
TEST_CASE("train_dense_2d_tensor__test_sparse_2d_tensor_"
          "spikeandslab_spikeandslab",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::spikeandslab, PriorTypes::spikeandslab})
      .runAndCheck(1898);
}

TEST_CASE("train_sparse_2d_tensor__test_sparse_2d_tensor_"
          "spikeandslab_spikeandslab",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::spikeandslab, PriorTypes::spikeandslab})
      .runAndCheck(1950);
}

//=================================================================

TEST_CASE("train_dense_2d_tensor__test_sparse_2d_tensor_"
          "normalone_normalone",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainDenseTensor2d, testSparseTensor2d, {PriorTypes::normalone, PriorTypes::normalone}).runAndCheck(2004);
}

TEST_CASE("train_sparse_2d_tensor__test_sparse_2d_tensor_"
          "normalone_normalone",
          TAG_TWO_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::normalone, PriorTypes::normalone}).runAndCheck(2056);
}

//=================================================================

TEST_CASE("train_dense_3d_tensor__test_sparse_3d_tensor_"
          "normal_normal",
          TAG_THREE_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainDenseTensor3d, testSparseTensor3d, {PriorTypes::normal, PriorTypes::normal, PriorTypes::normal})
      .runAndCheck(2110);
}

//=================================================================

TEST_CASE("train_dense_3d_tensor__test_sparse_3d_tensor_"
          "spikeandslab_spikeandslab",
          TAG_THREE_DIMENTIONAL_TENSOR_TESTS) {

  SmurffTest(trainDenseTensor3d, testSparseTensor3d,
             {PriorTypes::spikeandslab, PriorTypes::spikeandslab, PriorTypes::spikeandslab})
      .runAndCheck(2164);
}

//=================================================================

// not sure if this test produces correct results

TEST_CASE("train_dense_3d_tensor__test_sparse_3d_tensor_"
          "macau_normal__row_dense_side_info_none",
          TAG_THREE_DIMENTIONAL_TENSOR_TESTS) {
  SmurffTest(trainDenseTensor3d, testSparseTensor3d, {PriorTypes::macau, PriorTypes::normal, PriorTypes::normal})
      .addSideInfoConfig(0, rowSideDenseMatrix3d)
      .runAndCheck(2222);
}

//=================================================================

// not sure if this test produces correct results

TEST_CASE("train_dense_3d_tensor__test_sparse_3d_tensor_"
          "macauone_normal__row_dense_side_info_none",
          TAG_THREE_DIMENTIONAL_TENSOR_TESTS "[!mayfail]") {
  SmurffTest(trainDenseTensor3d, testSparseTensor3d, {PriorTypes::macauone, PriorTypes::normal, PriorTypes::normal})
      .addSideInfoConfig(0, rowSideDenseMatrix3d)
      .runAndCheck(2280);
}

} // namespace test
} // namespace smurff
