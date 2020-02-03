#pragma once

#include <limits>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>

#include <SmurffCpp/Configs/MatrixConfig.h>
#include <SmurffCpp/Configs/NoiseConfig.h>

#include <Utils/Error.h>

namespace smurff { namespace matrix_utils {
   // Conversion of MatrixConfig to/from sparse eigen matrix
   SparseMatrix sparse_to_eigen(const smurff::TensorConfig& matrixConfig);
   std::shared_ptr<smurff::MatrixConfig> eigen_to_sparse(const SparseMatrix &, smurff::NoiseConfig n = smurff::NoiseConfig(), bool isScarce = false);

   // Conversion of dense data to/from dense eigen matrix
   Matrix dense_to_eigen(const smurff::TensorConfig& matrixConfig);
   std::shared_ptr<smurff::MatrixConfig> eigen_to_dense(const Matrix &, smurff::NoiseConfig n = smurff::NoiseConfig());

   std::ostream& operator << (std::ostream& os, const MatrixConfig& mc);

   bool equals(const Matrix& m1, const Matrix& m2, double precision = std::numeric_limits<double>::epsilon());
   bool equals_vector(const Vector& v1, const Vector& v2, double precision = std::numeric_limits<double>::epsilon() * 100);
}}
