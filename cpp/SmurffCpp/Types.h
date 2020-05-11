#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Utils/Tensor.h>

namespace smurff {
   typedef SMURFF_FLOAT_TYPE float_type;

   template<typename F> double approx_epsilon();
   template<> inline double approx_epsilon<float> () { return 0.01; }
   template<> inline double approx_epsilon<double> () { return std::numeric_limits<float>::epsilon()*100; }

   template<typename F> constexpr af::dtype af_type_templ();
   template<> inline constexpr af::dtype af_type_templ<float> ()  { return f32; }
   template<> inline constexpr af::dtype af_type_templ<double> () { return f64; }
   constexpr af::dtype af_type = af_type_templ<float_type>();


   typedef Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
   typedef Eigen::Matrix<float_type, 1, Eigen::Dynamic, Eigen::RowMajor> Vector;
   typedef Eigen::Array<float_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Array2D;
   typedef Eigen::Array<float_type, 1, Eigen::Dynamic, Eigen::RowMajor> Array1D;
   typedef Eigen::SparseMatrix<float_type, Eigen::RowMajor> SparseMatrix;
};
