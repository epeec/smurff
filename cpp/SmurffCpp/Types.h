#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace smurff {
#ifdef USE_FLOAT
   typedef float float_type;
   const double APPROX_EPSILON = std::numeric_limits<float>::epsilon()*10000;
#else
   typedef double float_type;
   const double APPROX_EPSILON = std::numeric_limits<float>::epsilon()*100;
#endif

   typedef Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic> Matrix;
   typedef Eigen::Matrix<float_type, Eigen::Dynamic, 1> Vector;
   typedef Eigen::Array<float_type, Eigen::Dynamic, Eigen::Dynamic> Array2D;
   typedef Eigen::Array<float_type, Eigen::Dynamic, 1> Array1D;
   typedef Eigen::SparseMatrix<float_type> SparseMatrix;
};
