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

#include "Tests.h"

namespace smurff { namespace test {

struct CompareTest
{
  Config matrixConfig, tensorConfig;

  CompareTest(const MatrixConfig &matrix_train, const MatrixConfig &matrix_test, 
             const TensorConfig &tensor_train, const TensorConfig &tensor_test,
             std::vector<PriorTypes> priors)
      : matrixConfig(genConfig(matrix_train, matrix_test, priors)) 
      , tensorConfig(genConfig(tensor_train, tensor_test, priors)) 
  {}

  CompareTest &addSideInfoConfig(int m, const MatrixConfig &c,  bool direct = true, double tol = 1e-6)
  {
      matrixConfig.addSideInfoConfig(m, makeSideInfoConfig(c, direct, tol));
      tensorConfig.addSideInfoConfig(m, makeSideInfoConfig(c, direct, tol));
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

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    "
          "--prior normal normal --aux-data none none"
          "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normal normal --aux-data none none",
          TAG_VS_TESTS) {
  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::normal}).runAndCheck();
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior normal normal --aux-data none none"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normal normal --aux-data none none",
          TAG_VS_TESTS) {
  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::normal}).runAndCheck();
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    "
          "--prior normal spikeandslab --aux-data none none"
          "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normal spikeandslab --aux-data none none",
          TAG_VS_TESTS) {
  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::spikeandslab}).runAndCheck();
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior normal spikeandslab --aux-data none none"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normal spikeandslab --aux-data none none",
          TAG_VS_TESTS) {
  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::spikeandslab}).runAndCheck();
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    "
          "--prior spikeandslab normal --aux-data none none"
          "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior spikeandslab normal --aux-data none none",
          TAG_VS_TESTS) {
  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d, {PriorTypes::spikeandslab, PriorTypes::normal}).runAndCheck();
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior spikeandslab normal --aux-data none none"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior spikeandslab normal --aux-data none none",
          TAG_VS_TESTS) {
  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d, {PriorTypes::spikeandslab, PriorTypes::normal}).runAndCheck();
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    "
          "--prior spikeandslab spikeandslab --aux-data none none"
          "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior spikeandslab spikeandslab --aux-data none none",
          TAG_VS_TESTS) {
  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d, {PriorTypes::spikeandslab, PriorTypes::spikeandslab}).runAndCheck();
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior spikeandslab spikeandslab --aux-data none none"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior spikeandslab spikeandslab --aux-data none none",
          TAG_VS_TESTS) {
  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d, {PriorTypes::spikeandslab, PriorTypes::spikeandslab}).runAndCheck();
}

//==========================================================================

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    "
          "--prior normal normalone --aux-data none none"
          "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normal normalone --aux-data none none",
          TAG_VS_TESTS) {
  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::normalone}).runAndCheck();
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior normal normalone --aux-data none none"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normal normalone --aux-data none none",
          TAG_VS_TESTS) {
  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::normalone}).runAndCheck();
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    "
          "--prior normalone normal --aux-data none none"
          "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normalone normal --aux-data none none",
          TAG_VS_TESTS) {
  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d, {PriorTypes::normalone, PriorTypes::normal}).runAndCheck();
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior normalone normal --aux-data none none"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normalone normal --aux-data none none",
          TAG_VS_TESTS) {
  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d, {PriorTypes::normalone, PriorTypes::normal}).runAndCheck();
}

//             2. dense matrix
//             2. sparse matrix
TEST_CASE("matrix vs 2D-tensor"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    "
          "--prior normalone normalone --aux-data none none"
          "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normalone normalone --aux-data none none",
          TAG_VS_TESTS) {
  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d, {PriorTypes::normalone, PriorTypes::normalone}).runAndCheck();
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior normalone normalone --aux-data none none"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior normalone normalone --aux-data none none",
          TAG_VS_TESTS) {
  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d, {PriorTypes::normalone, PriorTypes::normalone}).runAndCheck();
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

  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d, {PriorTypes::macau, PriorTypes::macau}).addSideInfoConfig(0, rowSideDenseMatrix).addSideInfoConfig(1, colSideDenseMatrix).runAndCheck();
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior macau macau --side-info <row_side_info_dense_matrix> "
          "<col_side_info_dense_matrix> --direct"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior macau macau --side-info <row_side_info_dense_matrix> "
          "<col_side_info_dense_matrix> --direct",
          TAG_VS_TESTS) {

  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d, {PriorTypes::macau, PriorTypes::macau}).addSideInfoConfig(0, rowSideDenseMatrix).addSideInfoConfig(1, colSideDenseMatrix).runAndCheck();
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior macauone macauone --side-info <row_side_info_dense_matrix> "
          "<col_side_info_dense_matrix> --direct"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    "
          "--prior macauone macauone --side-info <row_side_info_dense_matrix> "
          "<col_side_info_dense_matrix> --direct",
          TAG_VS_TESTS) {

  CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d, {PriorTypes::macauone, PriorTypes::macauone}).addSideInfoConfig(0, rowSideDenseMatrix).addSideInfoConfig(1, colSideDenseMatrix).runAndCheck();
}

TEST_CASE("matrix vs 2D-tensor"
          "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> "
          "--prior macauone macauone --side-info <row_side_info_dense_matrix> "
          "<col_side_info_dense_matrix> --direct"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    "
          "--prior macauone macauone --side-info <row_side_info_dense_matrix> "
          "<col_side_info_dense_matrix> --direct",
          TAG_VS_TESTS) {

  CompareTest(trainSparseMatrix, testSparseMatrix, trainSparseTensor2d, testSparseTensor2d, {PriorTypes::macauone, PriorTypes::macauone}).addSideInfoConfig(0, rowSideDenseMatrix).addSideInfoConfig(1, colSideDenseMatrix).runAndCheck();
}


} } //end namespace smurff::test
