#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace smurff {
   typedef double flt;
   typedef Eigen::Matrix<flt, Eigen::Dynamic, Eigen::Dynamic> Matrix;
   typedef Eigen::Matrix<flt, Eigen::Dynamic, 1> Vector;
   typedef Eigen::Array<flt, Eigen::Dynamic, Eigen::Dynamic> Array2D;
   typedef Eigen::Array<flt, Eigen::Dynamic, 1> Array1D;
   typedef Eigen::SparseMatrix<flt> SparseMatrix;
};
