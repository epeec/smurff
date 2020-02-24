#include <vector>

#include <SmurffCpp/Types.h>

namespace smurff {

class MatrixConfig;
class TensorConfig;
struct ResultItem;

namespace test {

// noise
extern smurff::NoiseConfig fixed_ncfg;
extern smurff::NoiseConfig sampled_ncfg;

// dense train data
extern smurff::Matrix trainDenseMatrix;
extern smurff::DenseTensor trainDenseTensor2d;
extern smurff::DenseTensor trainDenseTensor3d;

// sparse train data
extern smurff::SparseMatrix trainSparseMatrix;
extern smurff::SparseTensor trainSparseTensor2d;

// sparse test data
extern smurff::SparseMatrix testSparseMatrix;
extern smurff::SparseTensor testSparseTensor2d;
extern smurff::SparseTensor testSparseTensor3d;

// aux data
extern smurff::MatrixConfig rowAuxDense;
extern smurff::MatrixConfig colAuxDense;

// side info
extern smurff::MatrixConfig rowSideDenseMatrix;
extern smurff::MatrixConfig colSideDenseMatrix;
extern smurff::MatrixConfig rowSideSparseMatrix;
extern smurff::MatrixConfig colSideSparseMatrix;
extern smurff::MatrixConfig rowSideDenseMatrix3d;

void REQUIRE_RESULT_ITEMS(const std::vector<smurff::ResultItem> &actualResultItems,
                          const std::vector<smurff::ResultItem> &expectedResultItems);
SideInfoConfig makeSideInfoConfig(const MatrixConfig &mcfg, bool direct = true, double tol = 1e-6);

template <class Train, class Test> Config genConfig(const Train &train, const Test &test, std::vector<PriorTypes> priors) {
  Config config;
  config.setBurnin(50);
  config.setNSamples(50);
  config.setVerbose(false);
  config.setRandomSeed(1234);
  config.setNumThreads(1);
  config.setNumLatent(4);
  DataConfig train_data_config(train);
  train_data_config.setNoiseConfig(fixed_ncfg);
  config.setTrain(std::make_shared<DataConfig>(train_data_config));
  config.setTest(std::make_shared<DataConfig>(test));
  config.setPriorTypes(priors);
  return config;
}
} // namespace test
} // namespace smurff
