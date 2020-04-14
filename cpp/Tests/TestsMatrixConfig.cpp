#include "catch.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Configs/MatrixConfig.h>

namespace smurff {

static NoiseConfig fixed_ncfg(NoiseTypes::fixed);

TEST_CASE("MatrixConfig::MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, const std::vector<double>& values, const NoiseConfig& noiseConfig)")
{
   std::vector<double> actualMatrixConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   MatrixConfig actualMatrixConfig(3, 4, actualMatrixConfigValues, fixed_ncfg);
   Matrix actualMatrix = matrix_utils::dense_to_eigen(actualMatrixConfig);

   Matrix expectedMatrix(3, 4);
   expectedMatrix(0, 0) = 1; expectedMatrix(0, 1) = 2; expectedMatrix(0, 2) = 3; expectedMatrix(0, 3) = 4;
   expectedMatrix(1, 0) = 5; expectedMatrix(1, 1) = 6; expectedMatrix(1, 2) = 7; expectedMatrix(1, 3) = 8;
   expectedMatrix(2, 0) = 9; expectedMatrix(2, 1) = 10; expectedMatrix(2, 2) = 11; expectedMatrix(2, 3) = 12;

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, const std::vector<std::uint32_t>& rows, const std::vector<std::uint32_t>& cols, const std::vector<double>& values, const NoiseConfig& noiseConfig)")
{
   std::vector<std::uint32_t> actualMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2};
   std::vector<std::uint32_t> actualMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> actualMatrixConfigValues      = { 1, 2, 3, 4, 9, 10, 11, 12 };
   MatrixConfig actualMatrixConfig(3, 4, actualMatrixConfigRows, actualMatrixConfigCols, actualMatrixConfigValues, fixed_ncfg, false);
   SparseMatrix actualMatrix = matrix_utils::sparse_to_eigen(actualMatrixConfig);

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

TEST_CASE("MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, const std::vector<std::uint32_t>& rows, const std::vector<std::uint32_t>& cols, const NoiseConfig& noiseConfig)")
{
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

   std::vector<std::uint32_t> actualMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> actualMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   MatrixConfig actualMatrixConfig(3, 4, actualMatrixConfigRows, actualMatrixConfigCols, fixed_ncfg, false);
   SparseMatrix actualMatrix = matrix_utils::sparse_to_eigen(actualMatrixConfig);

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

} // end namespace smurff
