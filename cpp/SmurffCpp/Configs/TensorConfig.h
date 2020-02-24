#pragma once

#include <vector>
#include <iostream>
#include <memory>
#include <cstdint>
#include <algorithm>

#include <SmurffCpp/Utils/PVec.hpp>
#include <SmurffCpp/Configs/DataConfig.h>

namespace smurff
{
   class Data;
   class IDataWriter;
   class IDataCreator;
   class ConfigFile;

   class TensorConfig : public DataConfig
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

      // Dense double tensor constructors
      TensorConfig(const std::vector<std::uint64_t>& dims,
                   const double* values,
                   const NoiseConfig& noiseConfig, PVec<> pos = {});
      TensorConfig(const std::vector<std::uint64_t>& dims,
                   const std::vector<double> &values,
                   const NoiseConfig& noiseConfig, PVec<> pos = {}) 
       : TensorConfig(dims, values.data(), noiseConfig, pos) {}

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

      // Sparse binary tensor constructors
      TensorConfig(const std::vector<std::uint64_t>& dims,
                   std::uint64_t nnz,
                   const std::vector<const std::uint32_t *>& columns,
                   const NoiseConfig& noiseConfig,
                   bool isScarce, PVec<> pos = {});

      TensorConfig(const std::vector<std::uint64_t> &dims,
                   const std::vector<std::vector<std::uint32_t>> &columns,
                   const NoiseConfig &noiseConfig,
                   bool isScarce, PVec<> pos = {}) 
          : TensorConfig(dims, columns[0].size(), vec_to_ptr(columns), noiseConfig, isScarce, pos) {} 

   public:
      virtual ~TensorConfig();

   public:
      std::pair<PVec<>, double> get(std::uint64_t) const;
      void set(std::uint64_t, PVec<>, double);

      const std::vector<std::uint32_t>& getRows() const { return getColumn(0); }
      const std::vector<std::uint32_t>& getCols() const { return getColumn(1); }
      const std::vector<double>& getValues() const { return m_values; }
      const columns_type& getColumns() const { 
         THROWERROR_ASSERT_MSG(!isDense(), "Cannot get index-vector for dense TensorConfig");
         return m_columns;
      }
      const std::vector<std::uint32_t>& getColumn(int i) const { 
         THROWERROR_ASSERT_MSG(!isDense(), "Cannot get index-vector for dense TensorConfig");
         return m_columns[i];
      }

      std::vector<std::uint32_t>& getRows() { return getColumn(0); }
      std::vector<std::uint32_t>& getCols() { return getColumn(1); }
      std::vector<double>& getValues() { return m_values; }
      std::vector<std::uint32_t>& getColumn(int i)  { 
         THROWERROR_ASSERT_MSG(!isDense(), "Cannot get index-vector for dense TensorConfig");
         return m_columns[i];
      }
      columns_type& getColumns() { 
         THROWERROR_ASSERT_MSG(!isDense(), "Cannot get index-vector for dense TensorConfig");
         return m_columns;
      }
      
   public:
      static std::shared_ptr<TensorConfig> restore_tensor_config(const ConfigFile& reader, const std::string& sec_name);

   public:
      virtual std::shared_ptr<Data> create(std::shared_ptr<IDataCreator> creator) const override;

   public:
      virtual void write(std::shared_ptr<IDataWriter> writer) const override;

   public:
      void check() const;
   };
}
