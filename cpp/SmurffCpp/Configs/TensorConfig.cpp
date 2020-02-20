#include "TensorConfig.h"

#include <numeric>

#include <SmurffCpp/Utils/PVec.hpp>
#include <SmurffCpp/IO/IDataWriter.h>
#include <SmurffCpp/DataMatrices/IDataCreator.h>
#include <SmurffCpp/IO/GenericIO.h>
#include <SmurffCpp/IO/INIFile.h>
#include <SmurffCpp/Utils/HDF5.h>
#include <Utils/Error.h>
#include <Utils/StringUtils.h>

namespace smurff {

static const std::string NONE_VALUE = "none";
static const std::string POS_TAG = "pos";
static const std::string FILE_TAG = "file";
static const std::string SCARCE_TAG = "scarce";
static const std::string TYPE_TAG = "type";

TensorConfig::TensorConfig ( bool isDense
                           , bool isBinary
                           , bool isScarce
                           , std::uint64_t nmodes
                           , std::uint64_t nnz
                           , const NoiseConfig& noiseConfig
                           , PVec<> pos
                           )
   : DataConfig(isDense, isBinary, isScarce, std::vector<std::uint64_t>(nmodes), nnz, noiseConfig, pos)
   , m_columns(nmodes)
   , m_values()
{
  // check(); // can't check here -- because many things still empty
}


// Dense double tensor constructors
TensorConfig::TensorConfig( const std::vector<std::uint64_t>& dims
                          , const double* values
                          , const NoiseConfig& noiseConfig
                          , PVec<> pos
                          )
   : DataConfig(true, false, false, dims, std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::uint64_t>()), noiseConfig, pos)
   , m_columns()
   , m_values(values, values + m_nnz)
{
   check();
}

// Sparse double tensor constructors
TensorConfig::TensorConfig( const std::vector<std::uint64_t>& dims
                          , std::uint64_t nnz
                          , const std::vector<const std::uint32_t *>& columns
                          , const double* values
                          , const NoiseConfig& noiseConfig
                          , bool isScarce 
                          , PVec<> pos
                          )
   : DataConfig(false, false, isScarce, dims, nnz, noiseConfig, pos)
   , m_values(values, values + nnz)
{
   for(auto col : columns)
   {
      m_columns.push_back(std::vector<std::uint32_t>(col, col + nnz));
   }

   check();
}

// Sparse binary tensor constructors
TensorConfig::TensorConfig( const std::vector<std::uint64_t>& dims
                          , std::uint64_t nnz
                          , const std::vector<const std::uint32_t *>& columns
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          , PVec<> pos
                          )
   : DataConfig(false, true, isScarce, dims, nnz, noiseConfig, pos)
   , m_values(nnz, 1.)
{
   for(auto col : columns)
   {
      m_columns.push_back(std::vector<std::uint32_t>(col, col + nnz));
   }

   check();
}

TensorConfig::~TensorConfig()
{
}

void TensorConfig::check() const
{
   DataConfig::check();
   THROWERROR_ASSERT(getNModes() >= 2);
   THROWERROR_ASSERT(m_values.size() == m_nnz);

   if (isDense())
   {
      THROWERROR_ASSERT(!isScarce());
      THROWERROR_ASSERT(m_nnz == std::accumulate(m_dims.begin(), m_dims.end(), 1ULL, std::multiplies<std::uint64_t>()));
   }
}

//
// other methods
//

std::pair<PVec<>, double> TensorConfig::get(std::uint64_t pos) const
{
   THROWERROR_ASSERT(!isDense());

   double val = m_values.at(pos);
   PVec<> coords(getNModes());
   for (unsigned j = 0; j < getNModes(); ++j)
         coords[j] = m_columns[j][pos];

   return std::make_pair(PVec<>(coords), val);
}

void TensorConfig::set(std::uint64_t pos, PVec<> coords, double value)
{
    m_values[pos] = value;
    for(unsigned j=0; j<getNModes(); ++j) m_columns[j][pos] = coords[j];
}

std::shared_ptr<TensorConfig> TensorConfig::restore_tensor_config(const ConfigFile& cfg_file, const std::string& sec_name)
{
   //restore filename
   std::string filename = cfg_file.get(sec_name, FILE_TAG, NONE_VALUE);
   if (filename == NONE_VALUE)
      return std::shared_ptr<TensorConfig>();

   //restore type
   bool is_scarce = cfg_file.get(sec_name, TYPE_TAG, SCARCE_TAG) == SCARCE_TAG;

   //restore data
   auto cfg = generic_io::read_data_config(filename, is_scarce);

   //restore instance
   cfg->restore(cfg_file, sec_name);

   return cfg;
}

std::shared_ptr<Data> TensorConfig::create(std::shared_ptr<IDataCreator> creator) const
{
   return creator->create(shared_from_this());
}

void TensorConfig::write(std::shared_ptr<IDataWriter> writer) const
{
   writer->write(shared_from_this());
}

} // end namespace smurff
