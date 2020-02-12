
std::shared_ptr<MatrixConfig> trainDenseMatrix() {
  MatrixConfig trainMatrixConfig(3, 4, {1., 5., 9.,  2., 6., 10., 3., 7., 11., 4., 8., 12.}, fixed_ncfg);
  return std::make_shared<MatrixConfig>(trainMatrixConfig);
}

std::shared_ptr<TensorConfig> trainDenseTensor2d() {
  TensorConfig trainTensorConfig({3, 4}, {1., 5., 9.,  2., 6., 10., 3., 7., 11., 4., 8., 12.}, fixed_ncfg);
  return std::make_shared<TensorConfig>(trainTensorConfig);
}

std::shared_ptr<TensorConfig> trainDenseTensor3d() {
  TensorConfig trainTensorConfig({2, 3, 4}, {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8., 9.,  10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.}, fixed_ncfg);
  return std::make_shared<TensorConfig>(trainTensorConfig);
}

// sparse train data (matrix/tensor 2d)

std::shared_ptr<MatrixConfig> trainSparseMatrix() {
  MatrixConfig trainMatrixConfig(3, 4, {0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}, {1., 2., 3., 4., 9., 10., 11., 12.}, fixed_ncfg, true);
  return std::make_shared<MatrixConfig>(trainMatrixConfig);
}

std::shared_ptr<TensorConfig> trainSparseTensor2d() {
  TensorConfig trainTensorConfig({3, 4}, { {0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}}, {1., 2., 3., 4., 9., 10., 11., 12.}, fixed_ncfg, true);
  return std::make_shared<TensorConfig>(trainTensorConfig);
}

// sparse test data (matrix/tensor 2d/tensor 3d)

std::shared_ptr<MatrixConfig> testSparseMatrix() {
  MatrixConfig testMatrixConfig(3, 4, {0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}, {1., 2., 3., 4., 9., 10., 11., 12.}, fixed_ncfg, true);
  return std::make_shared<MatrixConfig>(testMatrixConfig);
}

std::shared_ptr<TensorConfig> testSparseTensor2d() {
  TensorConfig testTensorConfig({3, 4}, { {0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}}, {1., 2., 3., 4., 9., 10., 11., 12.}, fixed_ncfg, true);
  return std::make_shared<TensorConfig>(testTensorConfig);
}

std::shared_ptr<TensorConfig> testSparseTensor3d() {
  TensorConfig testTensorConfig({2, 3, 4}, { {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 2, 2, 2, 2}, {0, 1, 2, 3, 0, 1, 2, 3}}, {1., 2., 3., 4., 9., 10., 11., 12.}, fixed_ncfg, true);
  return std::make_shared<TensorConfig>(testTensorConfig);
}

// aux data

std::shared_ptr<MatrixConfig> rowAuxDense() {
  MatrixConfig rowAuxDataDenseMatrixConfig(3, 1, {1., 2., 3.}, fixed_ncfg);
  rowAuxDataDenseMatrixConfig.setPos(PVec<>({0, 1}));
  return std::make_shared<MatrixConfig>(rowAuxDataDenseMatrixConfig);
}

std::shared_ptr<MatrixConfig> colAuxDense() {
  MatrixConfig colAuxDataDenseMatrixConfig(1, 4, {1., 2., 3., 4.}, fixed_ncfg);
  colAuxDataDenseMatrixConfig.setPos(PVec<>({1, 0}));
  return std::make_shared<MatrixConfig>(colAuxDataDenseMatrixConfig);
}

// side info

static NoiseConfig sampled_nc = []() {
    NoiseConfig nc(NoiseTypes::sampled);
    nc.setPrecision(10.0);
    return nc;
}();

std::shared_ptr<MatrixConfig> rowSideDenseMatrix() {
  MatrixConfig rowSideInfoDenseMatrixConfig(3, 1, {1., 2., 3.}, sampled_nc);
  return std::make_shared<MatrixConfig>(rowSideInfoDenseMatrixConfig);
}

std::shared_ptr<MatrixConfig> colSideDenseMatrix() {
  MatrixConfig colSideInfoDenseMatrixConfig(4, 1, {1., 2., 3., 4.}, sampled_nc);
  return std::make_shared<MatrixConfig>(colSideInfoDenseMatrixConfig);
}

std::shared_ptr<MatrixConfig> rowSideSparseMatrix() {
  MatrixConfig rowSideInfoSparseMatrixConfig(3, 1, {0, 1, 2}, {0, 0, 0}, {1., 2., 3.}, sampled_nc, false);
  return std::make_shared<MatrixConfig>(rowSideInfoSparseMatrixConfig);
}

std::shared_ptr<MatrixConfig> colSideSparseMatrix() {
  MatrixConfig colSideInfoSparseMatrixConfig(4, 1, {0, 1, 2, 3}, {0, 0, 0, 0}, {1., 2., 3., 4.}, sampled_nc, false);
  return std::make_shared<MatrixConfig>(colSideInfoSparseMatrixConfig);
}

std::shared_ptr<MatrixConfig> rowSideDenseMatrix3d() {
  MatrixConfig rowSideInfoDenseMatrixConfig(2, 3, {1., 2., 3., 4., 5., 6.}, sampled_nc);
  return std::make_shared<MatrixConfig>(rowSideInfoDenseMatrixConfig);
}
