#include "catch.hpp"

#include <sstream>
#include <cstdio>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/IO/MatrixIO.h>

namespace smurff {

static NoiseConfig fixed_ncfg(NoiseTypes::fixed);

TEST_CASE("matrix_io/read_matrix | matrix_io/write_matrix | .ddm")
{
   std::string matrixFilename = "matrixConfig.ddm";

   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<double> matrixConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   std::shared_ptr<MatrixConfig> matrixConfig(new MatrixConfig(matrixConfigNRow
                            , matrixConfigNCol
                            , matrixConfigValues
                            , fixed_ncfg
                            ));

   matrix_io::write_matrix(matrixFilename, matrixConfig);
   std::shared_ptr<MatrixConfig> actualMatrixConfig = matrix_io::read_matrix(matrixFilename, false);
   Matrix actualMatrix = matrix_utils::dense_to_eigen(*actualMatrixConfig);

   Matrix expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_matrix | matrix_io/write_matrix | .csv")
{
   std::string matrixFilename = "matrixConfig.csv";

   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<double> matrixConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   std::shared_ptr<MatrixConfig> matrixConfig(new MatrixConfig(matrixConfigNRow
                            , matrixConfigNCol
                            , matrixConfigValues
                            , fixed_ncfg
                            ));

   matrix_io::write_matrix(matrixFilename, matrixConfig);

   std::shared_ptr<MatrixConfig> actualMatrixConfig = matrix_io::read_matrix(matrixFilename, false);
   Matrix actualMatrix = matrix_utils::dense_to_eigen(*actualMatrixConfig);

   Matrix expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_matrix | matrix_io/write_matrix | .sdm")
{
   std::string matrixFilename = "matrixConfig.sdm";

   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<std::uint32_t> matrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> matrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> matrixConfigValues      = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> matrixConfig(new MatrixConfig(matrixConfigNRow
                            , matrixConfigNCol
                            , matrixConfigRows
                            , matrixConfigCols
                            , matrixConfigValues
                            , fixed_ncfg
                            , false
                            ));

   matrix_io::write_matrix(matrixFilename, matrixConfig);
   std::shared_ptr<MatrixConfig> actualMatrixConfig = matrix_io::read_matrix(matrixFilename, false);
   SparseMatrix actualMatrix = matrix_utils::sparse_to_eigen(*actualMatrixConfig);

   SparseMatrix expectedMatrix(3, 4);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 2));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 3));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 4));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 9));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 10));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 11));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 12));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_matrix | matrix_io/write_matrix | .mtx")
{
   // read/write dense matrix
   {
      std::string matrixFilename = "denseMatrixConfig.mtx";

      std::uint64_t matrixConfigNRow = 3;
      std::uint64_t matrixConfigNCol = 4;
      std::vector<double> matrixConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
      std::shared_ptr<MatrixConfig> matrixConfig(new MatrixConfig(matrixConfigNRow
                               , matrixConfigNCol
                               , matrixConfigValues
                               , fixed_ncfg
                               ));

      matrix_io::write_matrix(matrixFilename, matrixConfig);

      std::shared_ptr<MatrixConfig> actualMatrixConfig = matrix_io::read_matrix(matrixFilename, false);
      Matrix actualMatrix = matrix_utils::dense_to_eigen(*actualMatrixConfig);

      Matrix expectedMatrix(3, 4);
      expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

      std::remove(matrixFilename.c_str());
      REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
   }

   // read/write sparse matrix
   {
      std::string matrixFilename = "sparseMatrixConfig.mtx";

      std::uint64_t matrixConfigNRow = 3;
      std::uint64_t matrixConfigNCol = 4;
      std::vector<std::uint32_t> matrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
      std::vector<std::uint32_t> matrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
      std::vector<double> matrixConfigValues      = { 1, 2, 3, 4, 9, 10, 11, 12 };
      std::shared_ptr<MatrixConfig> matrixConfig(new MatrixConfig(matrixConfigNRow
                              , matrixConfigNCol
                              , matrixConfigRows
                              , matrixConfigCols
                              , matrixConfigValues
                              , fixed_ncfg
                              , false
                              ));

      matrix_io::write_matrix(matrixFilename, matrixConfig);
      std::shared_ptr<MatrixConfig> actualMatrixConfig = matrix_io::read_matrix(matrixFilename, false);
      SparseMatrix actualMatrix = matrix_utils::sparse_to_eigen(*actualMatrixConfig);

      SparseMatrix expectedMatrix(3, 4);
      std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 2));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 3));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 4));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 9));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 10));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 11));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 12));
      expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

      std::remove(matrixFilename.c_str());
      REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
   }
}

TEST_CASE("matrix_io/read_matrix | matrix_io/write_matrix | .sbm")
{
   std::string matrixFilename = "matrixConfig.sbm";

   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<std::uint32_t> matrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> matrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::shared_ptr<MatrixConfig> matrixConfig(new MatrixConfig(matrixConfigNRow
                            , matrixConfigNCol
                            , matrixConfigRows
                            , matrixConfigCols
                            , fixed_ncfg
                            , false
                            ));

   matrix_io::write_matrix(matrixFilename, matrixConfig);
   std::shared_ptr<MatrixConfig> actualMatrixConfig = matrix_io::read_matrix(matrixFilename, false);
   SparseMatrix actualMatrix = matrix_utils::sparse_to_eigen(*actualMatrixConfig);

   SparseMatrix expectedMatrix(3, 4);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 1));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

// ===

TEST_CASE("matrix_io/read_matrix_market | matrix_io/write_matrix_market | dense")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<double> matrixConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   std::shared_ptr<MatrixConfig> matrixConfig(new MatrixConfig(matrixConfigNRow
                            , matrixConfigNCol
                            , matrixConfigValues
                            , fixed_ncfg
                            ));

   std::stringstream matrixStream;
   matrix_io::write_matrix_market(matrixStream, matrixConfig);
   std::shared_ptr<MatrixConfig> actualMatrixConfig = matrix_io::read_matrix_market(matrixStream, false);
   Matrix actualMatrix = matrix_utils::dense_to_eigen(*actualMatrixConfig);

   Matrix expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_matrix_market | matrix_io/write_matrix_market | sparse")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<std::uint32_t> matrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> matrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> matrixConfigValues      = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> matrixConfig(new MatrixConfig(matrixConfigNRow
                            , matrixConfigNCol
                            , matrixConfigRows
                            , matrixConfigCols
                            , matrixConfigValues
                            , fixed_ncfg
                            , false
                            ));

   std::stringstream matrixStream;
   matrix_io::write_matrix_market(matrixStream, matrixConfig);
   std::shared_ptr<MatrixConfig> actualMatrixConfig = matrix_io::read_matrix_market(matrixStream, false);
   SparseMatrix actualMatrix = matrix_utils::sparse_to_eigen(*actualMatrixConfig);

   SparseMatrix expectedMatrix(3, 4);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 2));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 3));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 4));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 9));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 10));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 11));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 12));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_matrix_market | matrix_io/write_matrix_market | sparse binary")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<std::uint32_t> matrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> matrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::shared_ptr<MatrixConfig> matrixConfig =
      std::make_shared<MatrixConfig>( matrixConfigNRow
                                    , matrixConfigNCol
                                    , matrixConfigRows
                                    , matrixConfigCols
                                    , fixed_ncfg
                                    , false
                                    );

   std::stringstream matrixStream;
   matrix_io::write_matrix_market(matrixStream, matrixConfig);
   std::shared_ptr<MatrixConfig> actualMatrixConfig = matrix_io::read_matrix_market(matrixStream, false);
   SparseMatrix actualMatrix = matrix_utils::sparse_to_eigen(*actualMatrixConfig);

   SparseMatrix expectedMatrix(3, 4);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 1));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_dense_float64_bin | matrix_io/write_dense_float64_bin")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<double> matrixConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
   std::shared_ptr<MatrixConfig> matrixConfig(new MatrixConfig(matrixConfigNRow
                            , matrixConfigNCol
                            , matrixConfigValues
                            , fixed_ncfg
                            ));

   std::stringstream matrixStream;
   matrix_io::write_dense_float64_bin(matrixStream, matrixConfig);
   std::shared_ptr<MatrixConfig> actualMatrixConfig = matrix_io::read_dense_float64_bin(matrixStream);
   Matrix actualMatrix = matrix_utils::dense_to_eigen(*actualMatrixConfig);

   Matrix expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_dense_float64_csv | matrix_io/write_dense_float64_csv")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<double> matrixConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
   std::shared_ptr<MatrixConfig> matrixConfig(new MatrixConfig(matrixConfigNRow
                            , matrixConfigNCol
                            , matrixConfigValues
                            , fixed_ncfg
                            ));

   std::stringstream matrixConfigStream;
   matrix_io::write_dense_float64_csv(matrixConfigStream, matrixConfig);
   std::shared_ptr<MatrixConfig> actualMatrixConfig = matrix_io::read_dense_float64_csv(matrixConfigStream);
   Matrix actualMatrix = matrix_utils::dense_to_eigen(*actualMatrixConfig);

   Matrix expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_sparse_float64_bin | matrix_io/write_sparse_float64_bin")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<std::uint32_t> matrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> matrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> matrixConfigValues      = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> matrixConfig(new MatrixConfig(matrixConfigNRow
                            , matrixConfigNCol
                            , matrixConfigRows
                            , matrixConfigCols
                            , matrixConfigValues
                            , fixed_ncfg
                            , false
                            ));

   std::stringstream matrixConfigStream;
   matrix_io::write_sparse_float64_bin(matrixConfigStream, matrixConfig);
   std::shared_ptr<MatrixConfig> actualMatrixConfig = matrix_io::read_sparse_float64_bin(matrixConfigStream, false);
   SparseMatrix actualMatrix = matrix_utils::sparse_to_eigen(*actualMatrixConfig);

   SparseMatrix expectedMatrix(3, 4);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 2));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 3));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 4));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 9));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 10));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 11));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 12));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/read_sparse_binary_bin | matrix_io/write_sparse_binary_bin")
{
   std::uint64_t matrixConfigNRow = 3;
   std::uint64_t matrixConfigNCol = 4;
   std::vector<std::uint32_t> matrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> matrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::shared_ptr<MatrixConfig> matrixConfig(new MatrixConfig(matrixConfigNRow
                            , matrixConfigNCol
                            , matrixConfigRows
                            , matrixConfigCols
                            , fixed_ncfg
                            , false
                            ));

   std::stringstream matrixConfigStream;
   matrix_io::write_sparse_binary_bin(matrixConfigStream, matrixConfig);
   std::shared_ptr<MatrixConfig> actualMatrixConfig = matrix_io::read_sparse_binary_bin(matrixConfigStream, false);
   SparseMatrix actualMatrix = matrix_utils::sparse_to_eigen(*actualMatrixConfig);

   SparseMatrix expectedMatrix(3, 4);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 1));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

// ===

TEST_CASE("matrix_io/eigen::read_matrix(const std::string& filename, Vector& V) | matrix_io/eigen::write_matrix(const std::string& filename, const Matrix& X) | .ddm")
{
   std::string matrixFilename = "eigenVector.ddm";

   Vector expectedMatrix(3);
   expectedMatrix(0) = 1;
   expectedMatrix(1) = 4;
   expectedMatrix(2) = 9;
   matrix_io::eigen::write_matrix(matrixFilename, expectedMatrix);

   Vector actualMatrix;
   matrix_io::eigen::read_matrix(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/eigen::read_matrix(const std::string& filename, Vector& V) | matrix_io/eigen::write_matrix(const std::string& filename, const Matrix& X) | .csv")
{
   std::string matrixFilename = "eigenVector.csv";

   Vector expectedMatrix(3);
   expectedMatrix(0) = 1;
   expectedMatrix(1) = 4;
   expectedMatrix(2) = 9;
   matrix_io::eigen::write_matrix(matrixFilename, expectedMatrix);

   Vector actualMatrix;
   matrix_io::eigen::read_matrix(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

// ===

TEST_CASE("matrix_io/eigen::read_matrix(const std::string& filename, Matrix& X) | matrix_io/eigen::write_matrix(const std::string& filename, const Matrix& X) | .ddm")
{
   std::string matrixFilename = "denseEigenMatrix.ddm";

   Matrix expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
   matrix_io::eigen::write_matrix(matrixFilename, expectedMatrix);

   Matrix actualMatrix;
   matrix_io::eigen::read_matrix(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/eigen::read_matrix(const std::string& filename, Matrix& X) | matrix_io/eigen::write_matrix(const std::string& filename, const Matrix& X) | .csv")
{
   std::string matrixFilename = "denseEigenMatrix.csv";

   Matrix expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
   matrix_io::eigen::write_matrix(matrixFilename, expectedMatrix);

   Matrix actualMatrix;
   matrix_io::eigen::read_matrix(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

// ===

TEST_CASE("matrix_io/eigen::read_matrix(const std::string& filename, SparseMatrix& X) | matrix_io/eigen::write_matrix(const std::string& filename, const SparseMatrix& X) | .sdm")
{
   std::string matrixFilename = "sparseEigenMatrix.sdm";

   SparseMatrix expectedMatrix(3, 4);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 2));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 3));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 4));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 9));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 10));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 11));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 12));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   matrix_io::eigen::write_matrix(matrixFilename, expectedMatrix);

   SparseMatrix actualMatrix;
   matrix_io::eigen::read_matrix(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/eigen::read_matrix(const std::string& filename, SparseMatrix& X) | matrix_io/eigen::write_matrix(const std::string& filename, const SparseMatrix& X) | .mtx")
{
   // read/write dense matrix
   {
      std::string matrixFilename = "denseEigenMatrix.mtx";

      Matrix expectedMatrix(3, 4);
      expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
      matrix_io::eigen::write_matrix(matrixFilename, expectedMatrix);

      Matrix actualMatrix;
      matrix_io::eigen::read_matrix(matrixFilename, actualMatrix);

      std::remove(matrixFilename.c_str());
      REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
   }

   // read/write sparse matrix
   {
      std::string matrixFilename = "sparseEigenMatrix.mtx";

      SparseMatrix expectedMatrix(3, 4);
      std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 2));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 3));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 4));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 9));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 10));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 11));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 12));
      expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

      matrix_io::eigen::write_matrix(matrixFilename, expectedMatrix);

      SparseMatrix actualMatrix;
      matrix_io::eigen::read_matrix(matrixFilename, actualMatrix);

      std::remove(matrixFilename.c_str());
      REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
   }
}

TEST_CASE("matrix_io/eigen::read_matrix(const std::string& filename, SparseMatrix& X) | matrix_io/eigen::write_matrix(const std::string& filename, const SparseMatrix& X) | .sbm")
{
   std::string matrixFilename = "sparseEigenMatrix.sbm";

   SparseMatrix expectedMatrix(3, 4);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 1));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   matrix_io::eigen::write_matrix(matrixFilename, expectedMatrix);

   SparseMatrix actualMatrix;
   matrix_io::eigen::read_matrix(matrixFilename, actualMatrix);

   std::remove(matrixFilename.c_str());
   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("matrix_io/eigen::read_matrix | matrix_io/eigen::write_matrix | exception handling")
{
   // Read dense matrix as SparseMatrix
   {
      std::string matrixFilename = "denseMatrixMarket.mtx";

      Matrix expectedMatrix(3, 4);
      expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
      matrix_io::eigen::write_matrix(matrixFilename, expectedMatrix);

      SparseMatrix actualMatrix;
      REQUIRE_THROWS(matrix_io::eigen::read_matrix(matrixFilename, actualMatrix));
      std::remove(matrixFilename.c_str());
   }

   // Read sparse matrix as Matrix
   {
      std::string matrixFilename = "sparseMatrixMarket.mtx";

      SparseMatrix expectedMatrix(3, 4);
      std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 2));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 3));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 4));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 9));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 10));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 11));
      expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 12));
      expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());
      matrix_io::eigen::write_matrix(matrixFilename, expectedMatrix);

      Matrix actualMatrix;
      REQUIRE_THROWS(matrix_io::eigen::read_matrix(matrixFilename, actualMatrix));
      std::remove(matrixFilename.c_str());
   }
}

TEST_CASE("Genereate matrices for Python matrix_io tests", "[!hide]")
{
   std::uint64_t denseMatrixConfigNRow = 3;
   std::uint64_t denseMatrixConfigNCol = 4;
   std::vector<double> denseMatrixConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   std::shared_ptr<MatrixConfig> denseMatrixConfig =
      std::make_shared<MatrixConfig>( denseMatrixConfigNRow
                                    , denseMatrixConfigNCol
                                    , denseMatrixConfigValues
                                    , fixed_ncfg
                                    );
   matrix_io::write_matrix("cpp_generated_dense_matrix.ddm", denseMatrixConfig);
   matrix_io::write_matrix("cpp_generated_dense_matrix.mtx", denseMatrixConfig);
   matrix_io::write_matrix("cpp_generated_dense_matrix.csv", denseMatrixConfig);


   std::uint64_t sparseMatrixConfigNRow = 3;
   std::uint64_t sparseMatrixConfigNCol = 4;
   std::vector<std::uint32_t> sparseMatrixConfigRows = { 0, 0, 0, 0, 2,  2,  2,  2 };
   std::vector<std::uint32_t> sparseMatrixConfigCols = { 0, 1, 2, 3, 0,  1,  2,  3 };
   std::vector<double> sparseMatrixConfigValues      = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> sparseMatrixConfig =
      std::make_shared<MatrixConfig>( sparseMatrixConfigNRow
                                    , sparseMatrixConfigNCol
                                    , sparseMatrixConfigRows
                                    , sparseMatrixConfigCols
                                    , sparseMatrixConfigValues
                                    , fixed_ncfg
                                    , false
                                    );
   matrix_io::write_matrix("cpp_generated_sparse_matrix.sdm", sparseMatrixConfig);
   matrix_io::write_matrix("cpp_generated_sparse_matrix.mtx", sparseMatrixConfig);


   std::uint64_t sparseBinaryMatrixConfigNRow = 3;
   std::uint64_t sparseBinaryMatrixConfigNCol = 4;
   std::vector<std::uint32_t> sparseBinaryMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> sparseBinaryMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::shared_ptr<MatrixConfig> sparseBinaryMatrixConfig =
      std::make_shared<MatrixConfig>( sparseBinaryMatrixConfigNRow
                                    , sparseBinaryMatrixConfigNCol
                                    , sparseBinaryMatrixConfigRows
                                    , sparseBinaryMatrixConfigCols
                                    , fixed_ncfg
                                    , false
                                    );
   matrix_io::write_matrix("cpp_generated_sparse_matrix.sbm", sparseBinaryMatrixConfig);
}
} // end namespace smurff
