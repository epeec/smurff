#pragma once

#include <limits>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>

#include <SmurffCpp/Configs/MatrixConfig.h>
#include <SmurffCpp/Configs/NoiseConfig.h>

#include <Utils/Error.h>

namespace smurff { namespace matrix_utils {
   struct sparse_vec_iterator
   {
   template<typename T>
   sparse_vec_iterator(const T& matrixConfig, int pos)
      : rows(matrixConfig.getRows()),
        cols(matrixConfig.getCols()),
        values(matrixConfig.getValues()),
        pos(pos) {}

   sparse_vec_iterator(
        const std::vector<std::uint32_t>& rows,
        const std::vector<std::uint32_t>& cols,
        const std::vector<double>& values,
        int pos)
      : rows(rows), cols(cols), values(values), pos(pos) {}


   const std::vector<std::uint32_t>& rows;
   const std::vector<std::uint32_t>& cols;
   const std::vector<double>& values;
   int pos;

   bool operator!=(const sparse_vec_iterator &other) const {
      THROWERROR_ASSERT(&rows == &other.rows);
      THROWERROR_ASSERT(&cols == &other.cols);
      THROWERROR_ASSERT(&values == &other.values);
      return pos != other.pos;
   }

   sparse_vec_iterator &operator++() { pos++; return *this; }

   typedef Eigen::Triplet<float_type> T;
   T v;

   T* operator->() {
      // also convert from 1-base to 0-base
      uint32_t row = rows[pos];
      uint32_t col = cols[pos];
      float_type val = values[pos];
      v = T(row, col, val);
      return &v;
   }
   };

   // Conversion of MatrixConfig to/from sparse eigen matrix
   SparseMatrix sparse_to_eigen(const smurff::TensorConfig& matrixConfig);
   SparseMatrix sparse_to_eigen(const smurff::SparseTensor& );
   std::shared_ptr<smurff::MatrixConfig> eigen_to_sparse(const SparseMatrix &, smurff::NoiseConfig n = smurff::NoiseConfig(), bool isScarce = false);

   // Conversion of dense data to/from dense eigen matrix
   Matrix dense_to_eigen(const smurff::TensorConfig& matrixConfig);
   Matrix dense_to_eigen(const smurff::DenseTensor& );
   std::shared_ptr<smurff::MatrixConfig> eigen_to_dense(const Matrix &, smurff::NoiseConfig n = smurff::NoiseConfig());

   std::ostream& operator << (std::ostream& os, const MatrixConfig& mc);

   bool equals(const Matrix& m1, const Matrix& m2, double precision = std::numeric_limits<double>::epsilon());
   bool equals_vector(const Vector& v1, const Vector& v2, double precision = std::numeric_limits<double>::epsilon() * 100);
}}
