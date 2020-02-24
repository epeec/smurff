#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Configs/MatrixConfig.h>
#include <SmurffCpp/Configs/NoiseConfig.h>
#include <SmurffCpp/Configs/TensorConfig.h>

#include "Tests.h"

namespace smurff {
namespace test {

// noise config for train and test
smurff::NoiseConfig fixed_ncfg(NoiseTypes::fixed);

// dense train data (matrix/tensor 2d/tensor 3d)
smurff::Tensor trainDense({ 3, 4 }, {1., 5., 9., 2., 6., 10., 3., 7., 11., 4., 8., 12.});
smurff::MatrixConfig trainDenseMatrix(3, 4, {1., 5., 9., 2., 6., 10., 3., 7., 11., 4., 8., 12.}, fixed_ncfg);
smurff::TensorConfig trainDenseTensor2d({3, 4}, {1., 5., 9., 2., 6., 10., 3., 7., 11., 4., 8., 12.}, fixed_ncfg);
smurff::TensorConfig trainDenseTensor3d({2, 3, 4}, {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                                                    13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.},
                                        fixed_ncfg);

// sparse train data (matrix/tensor 2d)
smurff::MatrixConfig trainSparseMatrix(3, 4, {0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3},
                                       {1., 2., 3., 4., 9., 10., 11., 12.}, fixed_ncfg, true);
smurff::TensorConfig trainSparseTensor2d({3, 4}, {{0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}},
                                         {1., 2., 3., 4., 9., 10., 11., 12.}, fixed_ncfg, true);

// sparse test data (matrix/tensor 2d/tensor 3d)
smurff::MatrixConfig testSparseMatrix(3, 4, {0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3},
                                      {1., 2., 3., 4., 9., 10., 11., 12.}, fixed_ncfg, true);
smurff::TensorConfig testSparseTensor2d({3, 4}, {{0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}},
                                        {1., 2., 3., 4., 9., 10., 11., 12.}, fixed_ncfg, true);
smurff::SparseTensor testSparse({3, 4}, {{0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}},
                                        {1., 2., 3., 4., 9., 10., 11., 12.});
smurff::TensorConfig testSparseTensor3d({2, 3, 4},
                                        {{0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}},
                                        {1., 2., 3., 4., 9., 10., 11., 12.}, fixed_ncfg, true);

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
