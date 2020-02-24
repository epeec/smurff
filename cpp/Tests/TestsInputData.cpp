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
smurff::NoiseConfig unused_ncfg(NoiseTypes::unused);

// dense train data (matrix/tensor 2d/tensor 3d)
smurff::DenseTensor trainDenseTensor2d({3, 4}, {1., 5., 9., 2., 6., 10., 3., 7., 11., 4., 8., 12.});
smurff::DenseTensor trainDenseTensor3d({2, 3, 4}, {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                                                    13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
smurff::Matrix trainDenseMatrix(matrix_utils::dense_to_eigen(trainDenseTensor2d));

// sparse train data (matrix/tensor 2d)
smurff::SparseTensor trainSparseTensor2d({3, 4}, {{0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}},
                                         {1., 2., 3., 4., 9., 10., 11., 12.});
smurff::SparseMatrix trainSparseMatrix(matrix_utils::sparse_to_eigen(trainSparseTensor2d));

// sparse test data (matrix/tensor 2d/tensor 3d)
smurff::SparseTensor testSparseTensor2d({3, 4}, {{0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}},
                                        {1., 2., 3., 4., 9., 10., 11., 12.});
smurff::SparseTensor testSparseTensor3d({2, 3, 4},
                                        {{0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}},
                                        {1., 2., 3., 4., 9., 10., 11., 12.});
smurff::SparseMatrix testSparseMatrix(matrix_utils::sparse_to_eigen(testSparseTensor2d));

// aux data
static smurff::Matrix rowAuxDenseMatrix = matrix_utils::dense_to_eigen(smurff::DenseTensor({3, 1}, {1., 2., 3.}));
static smurff::Matrix colAuxDenseMatrix = matrix_utils::dense_to_eigen(smurff::DenseTensor({1, 4}, {1., 2., 3., 4.}));

smurff::DataConfig rowAuxDense(rowAuxDenseMatrix, fixed_ncfg, {0, 1});
smurff::DataConfig colAuxDense(colAuxDenseMatrix, fixed_ncfg, {1, 0});

// side info
smurff::Matrix       rowSideDenseMatrix   = matrix_utils::dense_to_eigen(smurff::DenseTensor({ 3, 1 }, {1., 2., 3.}));
smurff::Matrix       colSideDenseMatrix   = matrix_utils::dense_to_eigen(smurff::DenseTensor({ 4, 1 }, {1., 2., 3., 4.}));
smurff::SparseMatrix rowSideSparseMatrix  = matrix_utils::sparse_to_eigen(smurff::SparseTensor({ 3, 1 }, { {0, 1, 2}, {0, 0, 0} }, {1., 2., 3.}));
smurff::SparseMatrix colSideSparseMatrix  = matrix_utils::sparse_to_eigen(smurff::SparseTensor({ 4, 1 }, { {0, 1, 2, 3}, {0, 0, 0, 0} }, {1., 2., 3., 4.}));
smurff::Matrix       rowSideDenseMatrix3d = matrix_utils::dense_to_eigen(smurff::DenseTensor({ 2, 3 }, {1., 2., 3., 4., 5., 6.}));

} // namespace test
} // namespace smurff
