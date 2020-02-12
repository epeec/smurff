#pragma once

#include <vector>
#include <iostream>
#include <memory>

#include "TensorConfig.h"
#include "NoiseConfig.h"

namespace smurff
{
class Data;

class MatrixConfig : public TensorConfig
{
public:
   // Empty c'tor for filling later
   MatrixConfig(bool isDense, bool isBinary, bool isScarce, std::uint64_t nrow, std::uint64_t ncol, std::uint64_t nnz, const NoiseConfig &noiseConfig, PVec<> = {0,0});

   // Dense double matrix constructos
   MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, const double *values, const NoiseConfig &noiseConfig, PVec<> = {0,0});
   MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, const std::vector<double> values, const NoiseConfig &noiseConfig, PVec<> pos = {0,0})
       : MatrixConfig(nrow, ncol, values.data(), noiseConfig, pos) {}

   // Sparse double matrix constructors
   MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, std::uint64_t nnz, const std::uint32_t *rows, const std::uint32_t *cols, const double *values, const NoiseConfig &noiseConfig, bool isScarce, PVec<> = {0,0});
   MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, const std::vector<std::uint32_t> &rows, const std::vector<std::uint32_t> &cols, const std::vector<double> values, const NoiseConfig &noiseConfig, bool isScarce, PVec<> pos = {0,0})
       : MatrixConfig(nrow, ncol, values.size(), rows.data(), cols.data(), values.data(), noiseConfig, isScarce, pos) {}

   // Sparse binary matrix constructors
   MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, std::uint64_t nnz, const std::uint32_t *rows, const std::uint32_t *cols, const NoiseConfig &noiseConfig, bool isScarce, PVec<> = {0,0});
   MatrixConfig(std::uint64_t nrow, std::uint64_t ncol, const std::vector<std::uint32_t> &rows, const std::vector<std::uint32_t> &cols, const NoiseConfig &noiseConfig, bool isScarce, PVec<> pos = {0,0})
       : MatrixConfig(nrow, ncol, rows.size(), rows.data(), cols.data(), noiseConfig, isScarce, pos) {}

   MatrixConfig();

   std::shared_ptr<Data> create(std::shared_ptr<IDataCreator> creator) const override;

   void write(std::shared_ptr<IDataWriter> writer) const override;
};
} // namespace smurff
