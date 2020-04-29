#pragma once

#include <SmurffCpp/Utils/Tensor.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <arrayfire.h>

namespace smurff {
   typedef SMURFF_FLOAT_TYPE float_type;

   template<typename F> double approx_epsilon();
   template<> inline double approx_epsilon<float> () { return 0.01; }
   template<> inline double approx_epsilon<double> () { return std::numeric_limits<float>::epsilon()*100; }

   constexpr auto af_type = af::dtype_traits<float_type>::af_type;

   typedef Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
   typedef Eigen::Matrix<float_type, 1, Eigen::Dynamic, Eigen::RowMajor> Vector;
   typedef Eigen::Array<float_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Array2D;
   typedef Eigen::Array<float_type, 1, Eigen::Dynamic, Eigen::RowMajor> Array1D;
   typedef Eigen::SparseMatrix<float_type, Eigen::RowMajor> SparseMatrix;
};
