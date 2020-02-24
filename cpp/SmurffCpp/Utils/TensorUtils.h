#pragma once

#include <array>
#include <unordered_map>

#include <SmurffCpp/Types.h>

#include <SmurffCpp/Configs/TensorConfig.h>
#include <SmurffCpp/Configs/MatrixConfig.h>

namespace smurff { namespace tensor_utils {

// Print tensor config to console
std::ostream& operator << (std::ostream& os, const TensorConfig& tc);

// Take a matrix slice of tensor by fixing specific dimensions
Matrix slice(const TensorConfig& tensorConfig
   , const std::array<std::uint64_t, 2>& fixedDims
   , const std::unordered_map<std::uint64_t, std::uint32_t>& dimCoords
    );

   // Conversion of TensorConfig to SparseTensor
   SparseTensor sparse_to_tensor(const smurff::TensorConfig& tensorConfig);

   // Conversion of TensorConfig to DenseTensor
   Tensor dense_to_tensor(const smurff::TensorConfig& tensorConfig);

   // Conversion of Tensor to Matrix
   Tensor dense_to_tensor(const smurff::TensorConfig& tensorConfig);

}}
