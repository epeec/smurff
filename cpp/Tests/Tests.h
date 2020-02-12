#include <vector>

namespace smurff {

    class MatrixConfig;
    class TensorConfig;
    struct ResultItem;

    namespace test {

        // dense train data 
        extern smurff::MatrixConfig trainDenseMatrix;
        extern smurff::TensorConfig trainDenseTensor2d;
        extern smurff::TensorConfig trainDenseTensor3d;

        // sparse train data 
        extern smurff::MatrixConfig trainSparseMatrix;
        extern smurff::TensorConfig trainSparseTensor2d;

        // sparse test data 
        extern smurff::MatrixConfig testSparseMatrix;
        extern smurff::TensorConfig testSparseTensor2d;
        extern smurff::TensorConfig testSparseTensor3d;

        // aux data
        extern smurff::MatrixConfig rowAuxDense;
        extern smurff::MatrixConfig colAuxDense;

        // side info
        extern smurff::MatrixConfig rowSideDenseMatrix;
        extern smurff::MatrixConfig colSideDenseMatrix;
        extern smurff::MatrixConfig rowSideSparseMatrix;
        extern smurff::MatrixConfig colSideSparseMatrix;
        extern smurff::MatrixConfig rowSideDenseMatrix3d;

        void REQUIRE_RESULT_ITEMS(const std::vector<smurff::ResultItem> &actualResultItems, const std::vector<smurff::ResultItem> &expectedResultItems);
        std::shared_ptr<SideInfoConfig> makeSideInfoConfig(const MatrixConfig &mcfg, bool direct = true, double tol = 1e-6);

        template <class C>
        Config genConfig(const C &train, const C &test, std::vector<PriorTypes> priors)
        {
            Config config;
            config.setBurnin(50);
            config.setNSamples(50);
            config.setVerbose(false);
            config.setRandomSeed(1234);
            config.setNumThreads(1);
            config.setNumLatent(4);
            config.setTrain(std::make_shared<C>(train));
            config.setTest(std::make_shared<C>(test));
            config.setPriorTypes(priors);
            return config;
        }
    } // namespace test
} // namespace smurff
