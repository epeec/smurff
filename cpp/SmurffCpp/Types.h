#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace smurff {
   typedef SMURFF_FLOAT_TYPE float_type;

   template<typename F> double approx_epsilon();
   template<> inline double approx_epsilon<float> () { return 0.01; }
   template<> inline double approx_epsilon<double> () { return std::numeric_limits<float>::epsilon()*100; }

#define APPROX_EPSILON (smurff::approx_epsilon<smurff::float_type>())

   typedef Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic> Matrix;
   typedef Eigen::Matrix<float_type, Eigen::Dynamic, 1> Vector;
   typedef Eigen::Array<float_type, Eigen::Dynamic, Eigen::Dynamic> Array2D;
   typedef Eigen::Array<float_type, Eigen::Dynamic, 1> Array1D;
   typedef Eigen::SparseMatrix<float_type> SparseMatrix;
};
