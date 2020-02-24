#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Configs/MatrixConfig.h>
#include <SmurffCpp/Configs/NoiseConfig.h>
#include <SmurffCpp/Configs/TensorConfig.h>
#include <SmurffCpp/Utils/MatrixUtils.h>

#include "Tests.h"

namespace smurff {
namespace test {

// noise config for train and test
smurff::NoiseConfig fixed_ncfg(NoiseTypes::fixed);

// dense train data (matrix/tensor 2d/tensor 3d)
smurff::Matrix trainDenseMatrix(matrix_utils::dense_to_eigen(trainDenseTensor2d));
smurff::DenseTensor trainDenseTensor2d({3, 4}, {1., 5., 9., 2., 6., 10., 3., 7., 11., 4., 8., 12.});
smurff::DenseTensor trainDenseTensor3d({2, 3, 4}, {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                                                    13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});

// sparse train data (matrix/tensor 2d)
smurff::SparseMatrix trainSparseMatrix(matrix_utils::sparse_to_eigen(trainSparseTensor2d));
smurff::SparseTensor trainSparseTensor2d({3, 4}, {{0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}},
                                         {1., 2., 3., 4., 9., 10., 11., 12.});

// sparse test data (matrix/tensor 2d/tensor 3d)
smurff::SparseMatrix testSparseMatrix(matrix_utils::sparse_to_eigen(testSparseTensor2d));
smurff::SparseTensor testSparseTensor2d({3, 4}, {{0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}},
                                        {1., 2., 3., 4., 9., 10., 11., 12.});
smurff::SparseTensor testSparseTensor3d({2, 3, 4},
                                        {{0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}},
                                        {1., 2., 3., 4., 9., 10., 11., 12.});

// aux data
smurff::MatrixConfig rowAuxDense(3, 1, {1., 2., 3.}, fixed_ncfg, {0, 1});
smurff::MatrixConfig colAuxDense(1, 4, {1., 2., 3., 4.}, fixed_ncfg, {1, 0});

// noise config for sideinfo
smurff::NoiseConfig sampled_ncfg = []() {
  smurff::NoiseConfig nc(NoiseTypes::sampled);
  nc.setPrecision(10.0);
  return nc;
}();

// side info
smurff::MatrixConfig rowSideDenseMatrix(3, 1, {1., 2., 3.}, sampled_ncfg);
smurff::MatrixConfig colSideDenseMatrix(4, 1, {1., 2., 3., 4.}, sampled_ncfg);
smurff::MatrixConfig rowSideSparseMatrix(3, 1, {0, 1, 2}, {0, 0, 0}, {1., 2., 3.}, sampled_ncfg, false);
smurff::MatrixConfig colSideSparseMatrix(4, 1, {0, 1, 2, 3}, {0, 0, 0, 0}, {1., 2., 3., 4.}, sampled_ncfg, false);
smurff::MatrixConfig rowSideDenseMatrix3d(2, 3, {1., 2., 3., 4., 5., 6.}, sampled_ncfg);

} // namespace test
} // namespace smurff
