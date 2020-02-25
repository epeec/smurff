#include <numeric>

#include "TensorUtils.h"

#include <Utils/Error.h>

namespace smurff {

Matrix tensor_utils::slice( const TensorConfig& tensorConfig
                           , const std::array<std::uint64_t, 2>& fixedDims
                           , const std::unordered_map<std::uint64_t, std::uint32_t>& dimCoords)



// Conversion of TensorConfig to SparseTensor
SparseTensor tensor_utils::sparse_to_tensor(const smurff::TensorConfig& tensorConfig)
{
   return SparseTensor(tensorConfig.getDims(),  tensorConfig.getColumns(), tensorConfig.getValues());
}

// Conversion of TensorConfig to DenseTensor
DenseTensor tensor_utils::dense_to_tensor(const smurff::TensorConfig& tensorConfig)
{
   return DenseTensor{tensorConfig.getDims(),  tensorConfig.getValues()};
}

} // end namespace smurff
