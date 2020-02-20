#pragma once

#include <vector>
#include <iostream>
#include <memory>
#include <cstdint>
#include <algorithm>

namespace smurff
{
   class SparseTensor
   {
   public:
      typedef std::vector<std::vector<std::uint32_t>> columns_type;

   protected:
      columns_type         m_columns;
      std::vector<double>  m_values;

   private:
      std::vector<const std::uint32_t *> vec_to_ptr(const std::vector<std::vector<std::uint32_t>> &vec)
      {
         std::vector<const std::uint32_t *> ret;
         for(const auto &c : vec) ret.push_back(c.data());
         return ret;
      }

   public:
      // Empty c'tor for filling later
      TensorConfig(bool isDense, bool isBinary, bool isScarce,
                   std::uint64_t nmodes, std::uint64_t nnz, 
                   const NoiseConfig& noiseConfig, PVec<> pos = {});

      // Sparse double tensor constructors
      TensorConfig(const std::vector<std::uint64_t>& dims,
                   std::uint64_t nnz,
                   const std::vector<const std::uint32_t *>& columns,
                   const double* values,
                   const NoiseConfig& noiseConfig,
                   bool isScarce, PVec<> pos = {});

      TensorConfig(const std::vector<std::uint64_t> &dims,
                   const std::vector<std::vector<std::uint32_t>> &columns,
                   const std::vector<double> values,
                   const NoiseConfig &noiseConfig,
                   bool isScarce, PVec<> pos = {}) 
          : TensorConfig(dims, values.size(), vec_to_ptr(columns), values.data(), noiseConfig, isScarce, pos) {} 

      const std::vector<double>& getValues() const { return m_values; }
      const std::vector<std::uint32_t>& getColumn(int i) const { return m_columns.at(i); }

      std::vector<double>& getValues() { return m_values; }
      std::vector<std::uint32_t>& getColumn(int i)  { return m_columns.at(i); }
   };
}
