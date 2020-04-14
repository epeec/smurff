#include "catch.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/TensorUtils.h>
#include <SmurffCpp/Configs/TensorConfig.h>

namespace smurff {

static NoiseConfig fixed_ncfg(NoiseTypes::fixed);

TEST_CASE("TensorConfig(const std::vector<std::uint64_t>& dims, const std::vector<double> values, const NoiseConfig& noiseConfig)")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<double> tensorConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   TensorConfig tensorConfig(tensorConfigDims, tensorConfigValues.data(), fixed_ncfg);

   Matrix actualMatrix = matrix_utils::dense_to_eigen(tensorConfig);
   Matrix expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("TensorConfig(const std::vector<std::uint64_t>& dims, const std::vector<std::vector<std::uint32_t>>& columns, const std::vector<double>& values, const NoiseConfig& noiseConfig)")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<std::vector<std::uint32_t>> tensorConfigColumns = {
      { 0, 0, 0, 0, 2, 2, 2, 2 },
      { 0, 1, 2, 3, 0, 1, 2, 3 }
   };
   std::vector<double> tensorConfigValues = { 1, 2, 3, 4, 9, 10, 11, 12 };
   TensorConfig tensorConfig(tensorConfigDims, tensorConfigColumns, tensorConfigValues, fixed_ncfg, false);

   SparseMatrix actualMatrix = matrix_utils::sparse_to_eigen(tensorConfig);
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

TEST_CASE("TensorConfig(const std::vector<std::uint64_t>& dims, const std::vector<std::vector<std::uint32_t>>& columns, const NoiseConfig& noiseConfig)")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<std::vector<std::uint32_t>> tensorConfigColumns = { 
      { 0, 0, 0, 0, 2, 2, 2, 2 },
      { 0, 1, 2, 3, 0, 1, 2, 3 }
   };
   TensorConfig tensorConfig(tensorConfigDims, tensorConfigColumns, fixed_ncfg, false);

   SparseMatrix actualMatrix = matrix_utils::sparse_to_eigen(tensorConfig);
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

} // end namespace smurff