#pragma once

#include <array>
#include <unordered_map>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>

#include <SmurffCpp/Configs/TensorConfig.h>
#include <SmurffCpp/Configs/MatrixConfig.h>

namespace smurff { namespace tensor_utils {

Matrix dense_to_eigen(const smurff::TensorConfig& tensorConfig);


// Conversion of TensorConfig to sparse eigen matrix

SparseMatrix sparse_to_eigen(const smurff::TensorConfig& tensorConfig);

// Conversion of tensor config to matrix config

smurff::MatrixConfig tensor_to_matrix(const smurff::TensorConfig& tensorConfig);

// Print tensor config to console

std::ostream& operator << (std::ostream& os, const TensorConfig& tc);

// Take a matrix slice of tensor by fixing specific dimensions

Matrix slice(const TensorConfig& tensorConfig
   , const std::array<std::uint64_t, 2>& fixedDims
   , const std::unordered_map<std::uint64_t, std::uint32_t>& dimCoords
    );

}}
