#include "TensorConfig.h"

#include <numeric>

#include <SmurffCpp/Utils/PVec.hpp>
#include <SmurffCpp/IO/IDataWriter.h>
#include <SmurffCpp/DataMatrices/IDataCreator.h>
#include <SmurffCpp/IO/GenericIO.h>

#include <Utils/Error.h>
#include <Utils/StringUtils.h>

#define POS_TAG "pos"
#define FILE_TAG "file"
#define DENSE_TAG "dense"
#define SCARCE_TAG "scarce"
#define SPARSE_TAG "sparse"
#define TYPE_TAG "type"

#define NONE_TAG "none"

#define NOISE_MODEL_TAG "noise_model"
#define PRECISION_TAG "precision"
#define SN_INIT_TAG "sn_init"
#define SN_MAX_TAG "sn_max"
#define NOISE_THRESHOLD_TAG "noise_threshold"

namespace smurff {

TensorConfig::TensorConfig ( bool isDense
                           , bool isBinary
                           , bool isScarce
                           , std::uint64_t nmodes
                           , std::uint64_t nnz
                           , const NoiseConfig& noiseConfig
                           )
   : m_noiseConfig(noiseConfig)
   , m_isDense(isDense)
   , m_isBinary(isBinary)
   , m_isScarce(isScarce)
   , m_nmodes(nmodes)
   , m_dims()
   , m_nnz(nnz)
   , m_columns(nmodes)
   , m_values()
{
  // check(); // can't check here -- because many things still empty
}

// Dense double tensor constructors
TensorConfig::TensorConfig( const std::vector<std::uint64_t>& dims
                          , const double* values
                          , const NoiseConfig& noiseConfig
                          )
   : m_noiseConfig(noiseConfig)
   , m_isDense(true)
   , m_isBinary(false)
   , m_isScarce(false)
   , m_nmodes(dims.size())
   , m_dims(dims)
   , m_nnz(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::uint64_t>()))
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
                          )
   : m_noiseConfig(noiseConfig)
   , m_isDense(false)
   , m_isBinary(false)
   , m_isScarce(isScarce)
   , m_nmodes(dims.size())
   , m_dims(dims)
   , m_nnz(nnz)
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
                          )
   : m_noiseConfig(noiseConfig)
   , m_isDense(false)
   , m_isBinary(true)
   , m_isScarce(isScarce)
   , m_nmodes(dims.size())
   , m_dims(dims)
   , m_nnz(nnz)
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
   THROWERROR_ASSERT(m_nmodes > 0);
   THROWERROR_ASSERT(m_dims.size() == m_nmodes);
   THROWERROR_ASSERT(m_values.size() == m_nnz);

   if (isDense())
   {
      THROWERROR_ASSERT(m_columns.empty());
      THROWERROR_ASSERT(!isBinary());
      THROWERROR_ASSERT(m_nnz == std::accumulate(m_dims.begin(), m_dims.end(), 1ULL, std::multiplies<std::uint64_t>()));
   }
   else
   {
      THROWERROR_ASSERT(m_values.size() == m_nnz);
      THROWERROR_ASSERT(m_columns.size() == m_nmodes);
      for(int i=0; i<m_nmodes; ++i)
         THROWERROR_ASSERT(m_columns[i].size() == m_nnz);
   }
}

//
// other methods
//

bool TensorConfig::isDense() const
{
   return m_isDense;
}

bool TensorConfig::isBinary() const
{
   return m_isBinary;
}

bool TensorConfig::isScarce() const
{
   return m_isScarce;
}

std::uint64_t TensorConfig::getNNZ() const
{
   return m_nnz;
}

std::uint64_t TensorConfig::getNModes() const
{
   return m_nmodes;
}

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

const std::vector<std::uint64_t>& TensorConfig::getDims() const
{
   return m_dims;
}

const NoiseConfig& TensorConfig::getNoiseConfig() const
{
   return m_noiseConfig;
}

void TensorConfig::setNoiseConfig(const NoiseConfig& value)
{
   m_noiseConfig = value;
}

void TensorConfig::setFilename(const std::string &f)
{
    m_filename = f;
}

const std::string &TensorConfig::getFilename() const
{
    return m_filename;
}

void TensorConfig::setPos(const PVec<>& p)
{
   m_pos = std::make_shared<PVec<>>(p);
}

bool TensorConfig::hasPos() const
{
    return m_pos != nullptr;
}

const PVec<>& TensorConfig::getPos() const
{
    return *m_pos;
}

std::ostream& TensorConfig::info(std::ostream& os) const
{
   if (!m_dims.size())
   {
      os << "0";
   }
   else
   {
      os << m_dims.operator[](0);
      for (std::size_t i = 1; i < m_dims.size(); i++)
         os << " x " << m_dims.operator[](i);
   }
   if (getFilename().size())
   {
        os << " \"" << getFilename() << "\"";
   }
   if (hasPos())
   {
        os << " @[" << getPos() << "]";
   }
   return os;
}

std::string TensorConfig::info() const
{
    std::stringstream ss;
    info(ss);
    return ss.str();
}

void TensorConfig::save_tensor_config(INIFile& writer, const std::string& sec_name, int sec_idx, const std::shared_ptr<TensorConfig> &cfg)
{
   std::string section_name = (sec_idx >= 0) ? sec_name + "_" + std::to_string(sec_idx) : sec_name;
   
   if (cfg)
   {
      //save tensor config and noise config internally
      cfg->save(writer, section_name);
   }
   else
   {
      //save a placeholder since config can not serialize itself
      writer.startSection(section_name);
      writer.appendItem(section_name, FILE_TAG, NONE_TAG);
      writer.endSection();
   }
}

void TensorConfig::save(INIFile& writer, const std::string& section_name) const
{
   //write section name
   writer.startSection(section_name);

   //write tensor config position
   if (this->hasPos())
   {
      std::stringstream ss;
      ss << this->getPos();
      writer.appendItem(section_name, POS_TAG, ss.str());
   }

   //write tensor config filename
   writer.appendItem(section_name, FILE_TAG, this->getFilename());

   //write tensor config type
   std::string type_str = this->isDense() ? DENSE_TAG : this->isScarce() ? SCARCE_TAG : SPARSE_TAG;
   writer.appendItem(section_name, TYPE_TAG, type_str);

   //write noise config
   auto &noise_config = this->getNoiseConfig();
   if (noise_config.getNoiseType() != NoiseTypes::unset)
   {
      writer.appendItem(section_name, NOISE_MODEL_TAG, noiseTypeToString(noise_config.getNoiseType()));
      writer.appendItem(section_name, PRECISION_TAG, std::to_string(noise_config.getPrecision()));
      writer.appendItem(section_name, SN_INIT_TAG, std::to_string(noise_config.getSnInit()));
      writer.appendItem(section_name, SN_MAX_TAG, std::to_string(noise_config.getSnMax()));
      writer.appendItem(section_name, NOISE_THRESHOLD_TAG, std::to_string(noise_config.getThreshold()));
   }

   writer.endSection();
}

std::shared_ptr<TensorConfig> TensorConfig::restore_tensor_config(const INIFile& reader, const std::string& sec_name)
{
   //restore filename
   std::string filename = reader.get(sec_name, FILE_TAG, NONE_TAG);
   if (filename == NONE_TAG)
      return std::shared_ptr<TensorConfig>();

   //restore type
   bool is_scarce = reader.get(sec_name, TYPE_TAG, SCARCE_TAG) == SCARCE_TAG;

   //restore data
   auto cfg = generic_io::read_data_config(filename, is_scarce);

   //restore instance
   cfg->restore(reader, sec_name);

   return cfg;
}

bool TensorConfig::restore(const INIFile& reader, const std::string& sec_name)
{
   //restore position
   std::string pos_str = reader.get(sec_name, POS_TAG, NONE_TAG);
   if (pos_str != NONE_TAG)
   {
      std::vector<int> tokens;
      split(pos_str, tokens, ',');

      //assign position
      this->setPos(PVec<>(tokens));
   }

   //restore noise model
   NoiseConfig noise;

   NoiseTypes noiseType = stringToNoiseType(reader.get(sec_name, NOISE_MODEL_TAG, noiseTypeToString(NoiseTypes::unset)));
   if (noiseType != NoiseTypes::unset)
   {
      noise.setNoiseType(noiseType);
      noise.setPrecision(reader.getReal(sec_name, PRECISION_TAG, NoiseConfig::PRECISION_DEFAULT_VALUE));
      noise.setSnInit(reader.getReal(sec_name, SN_INIT_TAG, NoiseConfig::ADAPTIVE_SN_INIT_DEFAULT_VALUE));
      noise.setSnMax(reader.getReal(sec_name, SN_MAX_TAG, NoiseConfig::ADAPTIVE_SN_MAX_DEFAULT_VALUE));
      noise.setThreshold(reader.getReal(sec_name, NOISE_THRESHOLD_TAG, NoiseConfig::PROBIT_DEFAULT_VALUE));
   }

   //assign noise model
   this->setNoiseConfig(noise);

   return true;
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
