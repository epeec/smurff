#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace smurff {
   typedef FLOAT_TYPE float_type;

#if FLOAT_TYPE == float
   const double APPROX_EPSILON = std::numeric_limits<float>::epsilon()*10000;
#elif FLOAT_TYPE == double
   const double APPROX_EPSILON = std::numeric_limits<float>::epsilon()*100;
#else
   const double APPROX_EPSILON = std::numeric_limits<FLOAT_TYPE>::epsilon()*100;
#endif

   typedef Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic> Matrix;
   typedef Eigen::Matrix<float_type, Eigen::Dynamic, 1> Vector;
   typedef Eigen::Array<float_type, Eigen::Dynamic, Eigen::Dynamic> Array2D;
   typedef Eigen::Array<float_type, Eigen::Dynamic, 1> Array1D;
   typedef Eigen::SparseMatrix<float_type> SparseMatrix;
};
