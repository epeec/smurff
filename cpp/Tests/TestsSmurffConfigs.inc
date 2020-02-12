
std::shared_ptr<MatrixConfig> trainDenseMatrix() {
  std::vector<double> trainMatrixConfigVals = {1., 5., 9.,  2., 6., 10., 3., 7., 11., 4., 8., 12.};
  MatrixConfig trainMatrixConfig(3, 4, trainMatrixConfigVals, fixed_ncfg);
  return std::make_shared<MatrixConfig>(trainMatrixConfig);
}

std::shared_ptr<TensorConfig> trainDenseTensor2d() {
  std::vector<double> trainTensorConfigVals = {1., 5., 9.,  2., 6., 10.,
                                               3., 7., 11., 4., 8., 12.};
  TensorConfig trainTensorConfig(std::initializer_list<uint64_t>({3, 4}),
                                     trainTensorConfigVals.data(), fixed_ncfg);
  return std::make_shared<TensorConfig>(trainTensorConfig);
}

std::shared_ptr<TensorConfig> trainDenseTensor3d() {
  std::vector<double> trainTensorConfigVals = {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,
                                               9.,  10., 11., 12., 13., 14., 15., 16.,
                                               17., 18., 19., 20., 21., 22., 23., 24.};
  TensorConfig trainTensorConfig(std::initializer_list<uint64_t>({2, 3, 4}),
                                     trainTensorConfigVals, fixed_ncfg);
  return std::make_shared<TensorConfig>(trainTensorConfig);
}

// sparse train data (matrix/tensor 2d)

std::shared_ptr<MatrixConfig> trainSparseMatrix() {
  std::vector<std::uint32_t> trainMatrixConfigRows = {0, 0, 0, 0, 2, 2, 2, 2};
  std::vector<std::uint32_t> trainMatrixConfigCols = {0, 1, 2, 3, 0, 1, 2, 3};
  std::vector<double> trainMatrixConfigVals = {1., 2., 3., 4., 9., 10., 11., 12.};
  MatrixConfig trainMatrixConfig(3, 4, trainMatrixConfigRows,
                                     trainMatrixConfigCols,
                                     {1., 2., 3., 4., 9., 10., 11., 12.}, fixed_ncfg, true);
  return std::make_shared<MatrixConfig>(trainMatrixConfig);
}

std::shared_ptr<TensorConfig> trainSparseTensor2d() {
  std::vector<std::vector<std::uint32_t>> trainTensorConfigCols = {
      {0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}};
  std::vector<double> trainTensorConfigVals = {1., 2., 3., 4., 9., 10., 11., 12.};
  TensorConfig trainTensorConfig(std::initializer_list<uint64_t>({3, 4}),
                                     trainTensorConfigCols,
                                     trainTensorConfigVals, fixed_ncfg, true);
  return std::make_shared<TensorConfig>(trainTensorConfig);
}

// sparse test data (matrix/tensor 2d/tensor 3d)

std::shared_ptr<MatrixConfig> testSparseMatrix() {
  std::vector<std::uint32_t> testMatrixConfigRows = {0, 0, 0, 0, 2, 2, 2, 2};
  std::vector<std::uint32_t> testMatrixConfigCols = {0, 1, 2, 3, 0, 1, 2, 3};
  std::vector<double> testMatrixConfigVals = {1., 2., 3., 4., 9., 10., 11., 12.};
  MatrixConfig testMatrixConfig(3, 4, testMatrixConfigRows,
                                     testMatrixConfigCols, testMatrixConfigVals,
                                     fixed_ncfg, true);
  return std::make_shared<MatrixConfig>(testMatrixConfig);
}

std::shared_ptr<TensorConfig> testSparseTensor2d() {
  std::vector<std::vector<std::uint32_t>> testTensorConfigCols = {
      {0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}};
  std::vector<double> testTensorConfigVals = {1., 2., 3., 4., 9., 10., 11., 12.};
  TensorConfig testTensorConfig(std::initializer_list<uint64_t>({3, 4}),
                                     testTensorConfigCols, testTensorConfigVals,
                                     fixed_ncfg, true);
  return std::make_shared<TensorConfig>(testTensorConfig);
}

std::shared_ptr<TensorConfig> testSparseTensor3d() {
  std::vector<std::vector<std::uint32_t>> testTensorConfigCols = {
      {0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 2, 2, 2, 2},
      {0, 1, 2, 3, 0, 1, 2, 3}};
  std::vector<double> testTensorConfigVals = {1., 2., 3., 4., 9., 10., 11., 12.};
  TensorConfig testTensorConfig(std::initializer_list<uint64_t>({2, 3, 4}),
                                     testTensorConfigCols, testTensorConfigVals,
                                     fixed_ncfg, true);
  return std::make_shared<TensorConfig>(testTensorConfig);
}

// aux data

std::shared_ptr<MatrixConfig> rowAuxDense() {
  std::vector<double> rowAuxDataDenseMatrixConfigVals = {1., 2., 3.};
  MatrixConfig rowAuxDataDenseMatrixConfig(3, 1, rowAuxDataDenseMatrixConfigVals,
                                     fixed_ncfg);
  rowAuxDataDenseMatrixConfig.setPos(PVec<>({0, 1}));
  return std::make_shared<MatrixConfig>(rowAuxDataDenseMatrixConfig);
}

std::shared_ptr<MatrixConfig> colAuxDense() {
  std::vector<double> colAuxDataDenseMatrixConfigVals = {1., 2., 3., 4.};
  MatrixConfig colAuxDataDenseMatrixConfig(1, 4, colAuxDataDenseMatrixConfigVals,
                                     fixed_ncfg);
  colAuxDataDenseMatrixConfig.setPos(PVec<>({1, 0}));
  return std::make_shared<MatrixConfig>(colAuxDataDenseMatrixConfig);
}

// side info

std::shared_ptr<MatrixConfig> rowSideDenseMatrix() {
  NoiseConfig nc(NoiseTypes::sampled);
  nc.setPrecision(10.0);

  std::vector<double> rowSideInfoDenseMatrixConfigVals = {1., 2., 3.};
  MatrixConfig rowSideInfoDenseMatrixConfig(3, 1, rowSideInfoDenseMatrixConfigVals,
                                     nc);
  return std::make_shared<MatrixConfig>(rowSideInfoDenseMatrixConfig);
}

std::shared_ptr<MatrixConfig> colSideDenseMatrix() {
  NoiseConfig nc(NoiseTypes::sampled);
  nc.setPrecision(10.0);

  std::vector<double> colSideInfoDenseMatrixConfigVals = {1., 2., 3., 4.};
  MatrixConfig colSideInfoDenseMatrixConfig(4, 1, colSideInfoDenseMatrixConfigVals,
                                     nc);
  return std::make_shared<MatrixConfig>(colSideInfoDenseMatrixConfig);
}

std::shared_ptr<MatrixConfig> rowSideSparseMatrix() {
  NoiseConfig nc(NoiseTypes::sampled);
  nc.setPrecision(10.0);

  std::vector<std::uint32_t> rowSideInfoSparseMatrixConfigRows = {0, 1, 2};
  std::vector<std::uint32_t> rowSideInfoSparseMatrixConfigCols = {0, 0, 0};
  std::vector<double> rowSideInfoSparseMatrixConfigVals = {1., 2., 3.};
  MatrixConfig rowSideInfoSparseMatrixConfig(3, 1, rowSideInfoSparseMatrixConfigRows,
                                     rowSideInfoSparseMatrixConfigCols,
                                     rowSideInfoSparseMatrixConfigVals, nc,
                                     true);
  return std::make_shared<MatrixConfig>(rowSideInfoSparseMatrixConfig);
}

std::shared_ptr<MatrixConfig> colSideSparseMatrix() {
  NoiseConfig nc(NoiseTypes::sampled);
  nc.setPrecision(10.0);

  std::vector<std::uint32_t> colSideInfoSparseMatrixConfigRows = {0, 1, 2, 3};
  std::vector<std::uint32_t> colSideInfoSparseMatrixConfigCols = {0, 0, 0, 0};
  std::vector<double> colSideInfoSparseMatrixConfigVals = {1., 2., 3., 4.};
  MatrixConfig colSideInfoSparseMatrixConfig(4, 1, colSideInfoSparseMatrixConfigRows,
                                     colSideInfoSparseMatrixConfigCols,
                                     colSideInfoSparseMatrixConfigVals, nc,
                                     true);
  return std::make_shared<MatrixConfig>(colSideInfoSparseMatrixConfig);
}

std::shared_ptr<MatrixConfig> rowSideDenseMatrix3d() {
  NoiseConfig nc(NoiseTypes::sampled);
  nc.setPrecision(10.0);

  std::vector<double> rowSideInfoDenseMatrixConfigVals = {1., 2., 3., 4., 5., 6.};
  MatrixConfig rowSideInfoDenseMatrixConfig(2, 3, rowSideInfoDenseMatrixConfigVals,
                                     nc);
  return std::make_shared<MatrixConfig>(rowSideInfoDenseMatrixConfig);
}
