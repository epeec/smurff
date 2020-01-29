#pragma once

#include <vector>
#include <iostream>
#include <memory>
#include <cstdint>
#include <algorithm>

#include <SmurffCpp/Utils/PVec.hpp>
#include <SmurffCpp/Configs/NoiseConfig.h>
#include <SmurffCpp/IO/INIFile.h>

namespace smurff
{
   class Data;
   class IDataWriter;
   class IDataCreator;

   class TensorConfig : public std::enable_shared_from_this<TensorConfig>
   {
   public:
      typedef std::vector<std::vector<std::uint32_t>> columns_type;

   private:
      NoiseConfig m_noiseConfig;

   protected:
      bool m_isDense;
      bool m_isBinary;
      bool m_isScarce;

      std::uint64_t m_nmodes;
      std::vector<std::uint64_t> m_dims;
      std::uint64_t m_nnz;

      columns_type         m_columns;
      std::vector<double>  m_values;

   private:
      std::string m_filename;
      std::shared_ptr<PVec<>> m_pos;

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
                   const NoiseConfig& noiseConfig);

      // Dense double tensor constructors
      TensorConfig(const std::vector<std::uint64_t>& dims,
                   const double* values,
                   const NoiseConfig& noiseConfig);
      TensorConfig(const std::vector<std::uint64_t>& dims,
                   const std::vector<double> &values,
                   const NoiseConfig& noiseConfig) 
       : TensorConfig(dims, values.data(), noiseConfig) {}

      // Sparse double tensor constructors
      TensorConfig(const std::vector<std::uint64_t>& dims,
                   std::uint64_t nnz,
                   const std::vector<const std::uint32_t *>& columns,
                   const double* values,
                   const NoiseConfig& noiseConfig,
                   bool isScarce);

      TensorConfig(const std::vector<std::uint64_t> &dims,
                   const std::vector<std::vector<std::uint32_t>> &columns,
                   const std::vector<double> values,
                   const NoiseConfig &noiseConfig,
                   bool isScarce) 
          : TensorConfig(dims, values.size(), vec_to_ptr(columns), values.data(), noiseConfig, isScarce) {} 

      // Sparse binary tensor constructors
      TensorConfig(const std::vector<std::uint64_t>& dims,
                   std::uint64_t nnz,
                   const std::vector<const std::uint32_t *>& columns,
                   const NoiseConfig& noiseConfig,
                   bool isScarce);

      TensorConfig(const std::vector<std::uint64_t> &dims,
                   const std::vector<std::vector<std::uint32_t>> &columns,
                   const NoiseConfig &noiseConfig,
                   bool isScarce) 
          : TensorConfig(dims, columns[0].size(), vec_to_ptr(columns), noiseConfig, isScarce) {} 

   public:
      virtual ~TensorConfig();

   public:
      const NoiseConfig& getNoiseConfig() const;
      void setNoiseConfig(const NoiseConfig& value);

      bool isDense() const;
      bool isBinary() const;
      bool isScarce() const;

      std::uint64_t getNModes() const;
      std::uint64_t getNNZ() const;

      std::pair<PVec<>, double> get(std::uint64_t) const;
      void set(std::uint64_t, PVec<>, double);

      const std::vector<std::uint64_t>& getDims() const;
      std::uint64_t getNRow() const { return getDims().at(0); }
      std::uint64_t getNCol() const { return getDims().at(1); }

      const std::vector<std::uint32_t>& getRows() const { return getColumn(0); }
      const std::vector<std::uint32_t>& getCols() const { return getColumn(1); }
      const std::vector<double>& getValues() const { return m_values; }
      const std::vector<std::uint32_t>& getColumn(int i) const{ return m_columns[i]; }

      std::vector<std::uint32_t>& getRows() { return getColumn(0); }
      std::vector<std::uint32_t>& getCols() { return getColumn(1); }
      std::vector<double>& getValues() { return m_values; }
      std::vector<std::uint32_t>& getColumn(int i){ return m_columns[i]; } 

      void setFilename(const std::string& f);
      const std::string &getFilename() const;

      void setPos(const PVec<>& p);
      void setPos(const std::vector<int> &p) { setPos(PVec<>(p)); }
      bool hasPos() const;
      const PVec<> &getPos() const;

   public:
      virtual std::ostream& info(std::ostream& os) const;
      virtual std::string info() const;
      virtual void save(INIFile& writer, const std::string& section_name) const;
      virtual bool restore(const INIFile& reader, const std::string& sec_name);

   public:
      static void save_tensor_config(INIFile& writer, const std::string& sec_name, int sec_idx, const std::shared_ptr<TensorConfig> &cfg);

      static std::shared_ptr<TensorConfig> restore_tensor_config(const INIFile& reader, const std::string& sec_name);

   public:
      virtual std::shared_ptr<Data> create(std::shared_ptr<IDataCreator> creator) const;

   public:
      virtual void write(std::shared_ptr<IDataWriter> writer) const;
   };
}
