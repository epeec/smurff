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

namespace smurff {
namespace test {

struct CompareTest {
  Config matrixConfig, tensorConfig;

  CompareTest(const Matrix &matrix_train, const SparseMatrix &matrix_test,
              const DenseTensor &tensor_train, const SparseTensor &tensor_test,
              std::vector<PriorTypes> priors)
      : matrixConfig(genConfig(matrix_train, matrix_test, priors)),
        tensorConfig(genConfig(tensor_train, tensor_test, priors)) {}

  CompareTest(const SparseMatrix &matrix_train, const SparseMatrix &matrix_test,
              const SparseTensor &tensor_train, const SparseTensor &tensor_test,
              std::vector<PriorTypes> priors)
      : matrixConfig(genConfig(matrix_train, matrix_test, priors)),
        tensorConfig(genConfig(tensor_train, tensor_test, priors)) {}
 
  template<class M>
  CompareTest &addSideInfoConfig(int m, const M &c) {
    matrixConfig.addSideInfoConfig(m) = makeSideInfoConfig(c);
    tensorConfig.addSideInfoConfig(m) = makeSideInfoConfig(c);
    return *this;
  }

  void runAndCheck() {
    std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixConfig);
    std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorConfig);
    matrixSession->run();
    tensorSession->run();

    REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
    REQUIRE_RESULT_ITEMS(matrixSession->getResultItems(), tensorSession->getResultItems());
  }
};

#ifdef USE_BOOST_RANDOM
#define TAG_VS_TESTS "[versus][random]"
#else
#define TAG_VS_TESTS "[versus][random][!mayfail]"
#endif

//=================================================================

TEST_CASE("matrix_vs_2D-tensor_train_dense_matrix_test_sparse_matrix_normal_normal__none_none_train_dense_2d_tensor_test_sparse_2d_tensor_normal_normal__none_none",
          TAG_VS_TESTS) {
  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d,
              {PriorTypes::normal, PriorTypes::normal})
      .runAndCheck();
}

TEST_CASE("matrix_vs_2D-tensor_train_sparse_matrix_test_sparse_matrix_normal_normal__none_none_train_sparse_2d_tensor_test_sparse_2d_tensor_normal_normal__none_none",
          TAG_VS_TESTS) {
  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d,
              {PriorTypes::normal, PriorTypes::normal})
      .runAndCheck();
}

TEST_CASE("matrix_vs_2D-tensor_train_dense_matrix_test_sparse_matrix_normal_spikeandslab__none_none_train_dense_2d_tensor_test_sparse_2d_tensor_normal_spikeandslab__none_none",
          TAG_VS_TESTS) {
  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d,
              {PriorTypes::normal, PriorTypes::spikeandslab})
      .runAndCheck();
}

TEST_CASE("matrix_vs_2D-tensor_train_sparse_matrix_test_sparse_matrix_normal_spikeandslab__none_none_train_sparse_2d_tensor_test_sparse_2d_tensor_normal_spikeandslab__none_none",
          TAG_VS_TESTS) {
  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d,
              {PriorTypes::normal, PriorTypes::spikeandslab})
      .runAndCheck();
}

TEST_CASE("matrix_vs_2D-tensor_train_dense_matrix_test_sparse_matrix_spikeandslab_normal__none_none_train_dense_2d_tensor_test_sparse_2d_tensor_spikeandslab_normal__none_none",
          TAG_VS_TESTS) {
  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d,
              {PriorTypes::spikeandslab, PriorTypes::normal})
      .runAndCheck();
}

TEST_CASE("matrix_vs_2D-tensor_train_sparse_matrix_test_sparse_matrix_spikeandslab_normal__none_none_train_sparse_2d_tensor_test_sparse_2d_tensor_spikeandslab_normal__none_none",
          TAG_VS_TESTS) {
  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d,
              {PriorTypes::spikeandslab, PriorTypes::normal})
      .runAndCheck();
}

TEST_CASE("matrix_vs_2D-tensor_train_dense_matrix_test_sparse_matrix_spikeandslab_spikeandslab__none_none_train_dense_2d_tensor_test_sparse_2d_tensor_spikeandslab_spikeandslab__none_none",
          TAG_VS_TESTS) {
  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d,
              {PriorTypes::spikeandslab, PriorTypes::spikeandslab})
      .runAndCheck();
}

TEST_CASE("matrix_vs_2D-tensor_train_sparse_matrix_test_sparse_matrix_spikeandslab_spikeandslab__none_none_train_sparse_2d_tensor_test_sparse_2d_tensor_spikeandslab_spikeandslab__none_none",
          TAG_VS_TESTS) {
  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d,
              {PriorTypes::spikeandslab, PriorTypes::spikeandslab})
      .runAndCheck();
}

//==========================================================================

TEST_CASE("matrix_vs_2D-tensor_train_dense_matrix_test_sparse_matrix_normal_normalone__none_none_train_dense_2d_tensor_test_sparse_2d_tensor_normal_normalone__none_none",
          TAG_VS_TESTS) {
  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d,
              {PriorTypes::normal, PriorTypes::normalone})
      .runAndCheck();
}

TEST_CASE("matrix_vs_2D-tensor_train_sparse_matrix_test_sparse_matrix_normal_normalone__none_none_train_sparse_2d_tensor_test_sparse_2d_tensor_normal_normalone__none_none",
          TAG_VS_TESTS) {
  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d,
              {PriorTypes::normal, PriorTypes::normalone})
      .runAndCheck();
}

TEST_CASE("matrix_vs_2D-tensor_train_dense_matrix_test_sparse_matrix_normalone_normal__none_none_train_dense_2d_tensor_test_sparse_2d_tensor_normalone_normal__none_none",
          TAG_VS_TESTS) {
  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d,
              {PriorTypes::normalone, PriorTypes::normal})
      .runAndCheck();
}

TEST_CASE("matrix_vs_2D-tensor_train_sparse_matrix_test_sparse_matrix_normalone_normal__none_none_train_sparse_2d_tensor_test_sparse_2d_tensor_normalone_normal__none_none",
          TAG_VS_TESTS) {
  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d,
              {PriorTypes::normalone, PriorTypes::normal})
      .runAndCheck();
}

//             2. dense matrix
//             2. sparse matrix
TEST_CASE("matrix_vs_2D-tensor_train_dense_matrix_test_sparse_matrix_normalone_normalone__none_none_train_dense_2d_tensor_test_sparse_2d_tensor_normalone_normalone__none_none",
          TAG_VS_TESTS) {
  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d,
              {PriorTypes::normalone, PriorTypes::normalone})
      .runAndCheck();
}

TEST_CASE("matrix_vs_2D-tensor_train_sparse_matrix_test_sparse_matrix_normalone_normalone__none_none_train_sparse_2d_tensor_test_sparse_2d_tensor_normalone_normalone__none_none",
          TAG_VS_TESTS) {
  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d,
              {PriorTypes::normalone, PriorTypes::normalone})
      .runAndCheck();
}

//==========================================================================

TEST_CASE("matrix_vs_2D-tensor_train_dense_2d_tensor_test_sparse_2d_tensor_macau_macau__row_side_info_dense_matrix_col_side_info_dense_matrix_train_dense_matrix_test_sparse_matrix_macau_macau__row_side_info_dense_matrix_col_side_info_dense_matrix_",
          TAG_VS_TESTS) {

  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d,
              {PriorTypes::macau, PriorTypes::macau})
      .addSideInfoConfig(0, rowSideDenseMatrix)
      .addSideInfoConfig(1, colSideDenseMatrix)
      .runAndCheck();
}

TEST_CASE("matrix_vs_2D-tensor_train_sparse_2d_tensor_test_sparse_2d_tensor_macau_macau__row_side_info_dense_matrix_col_side_info_dense_matrix_train_sparse_matrix_test_sparse_matrix_macau_macau__row_side_info_dense_matrix_col_side_info_dense_matrix_",
          TAG_VS_TESTS) {

  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d,
              {PriorTypes::macau, PriorTypes::macau})
      .addSideInfoConfig(0, rowSideDenseMatrix)
      .addSideInfoConfig(1, colSideDenseMatrix)
      .runAndCheck();
}

TEST_CASE("matrix_vs_2D-tensor_train_dense_2d_tensor_test_sparse_2d_tensor_macauone_macauone__row_side_info_dense_matrix_col_side_info_dense_matrix_train_dense_matrix_test_sparse_matrix_macauone_macauone__row_side_info_dense_matrix_col_side_info_dense_matrix_",
          TAG_VS_TESTS) {

  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d,
              {PriorTypes::macauone, PriorTypes::macauone})
      .addSideInfoConfig(0, rowSideDenseMatrix)
      .addSideInfoConfig(1, colSideDenseMatrix)
      .runAndCheck();
}

TEST_CASE("matrix_vs_2D-tensor_train_sparse_2d_tensor_test_sparse_2d_tensor_macauone_macauone__row_side_info_dense_matrix_col_side_info_dense_matrix_train_sparse_matrix_test_sparse_matrix_macauone_macauone__row_side_info_dense_matrix_col_side_info_dense_matrix_",
          TAG_VS_TESTS) {

  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d,
              {PriorTypes::macauone, PriorTypes::macauone})
      .addSideInfoConfig(0, rowSideDenseMatrix)
      .addSideInfoConfig(1, colSideDenseMatrix)
      .runAndCheck();
}

} // namespace test
} // namespace smurff
