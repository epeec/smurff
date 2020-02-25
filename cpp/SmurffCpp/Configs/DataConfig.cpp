#include "DataConfig.h"

#include <numeric>

#include <SmurffCpp/Utils/PVec.hpp>
#include <SmurffCpp/DataMatrices/IDataCreator.h>
#include <SmurffCpp/Utils/ConfigFile.h>
#include <Utils/Error.h>
#include <Utils/StringUtils.h>

namespace smurff {

static const std::string POS_TAG = "pos";
static const std::string FILE_TAG = "file";
static const std::string DATA_TAG = "data";
static const std::string DENSE_TAG = "dense";
static const std::string SCARCE_TAG = "scarce";
static const std::string SPARSE_TAG = "sparse";
static const std::string TYPE_TAG = "type";

static const std::string NONE_VALUE("none");

static const std::string NOISE_MODEL_TAG = "noise_model";
static const std::string PRECISION_TAG = "precision";
static const std::string SN_INIT_TAG = "sn_init";
static const std::string SN_MAX_TAG = "sn_max";
static const std::string NOISE_THRESHOLD_TAG = "noise_threshold";

DataConfig::DataConfig ( const Matrix &m
                       , const NoiseConfig& noiseConfig
                       , PVec<> pos
                       )
   : m_noiseConfig(noiseConfig)
   , m_pos(pos)
{
   setData(m);
}

DataConfig::DataConfig ( const SparseMatrix &m
                       , bool isScarce
                       , const NoiseConfig& noiseConfig
                       , PVec<> pos
                       )
   : m_noiseConfig(noiseConfig)
   , m_pos(pos)
{
   setData(m, isScarce);
}

DataConfig::DataConfig ( const DenseTensor &m
                       , const NoiseConfig& noiseConfig
                       , PVec<> pos
                       )
   : m_noiseConfig(noiseConfig)
   , m_pos(pos)
{
   setData(m);
}

DataConfig::DataConfig ( const SparseTensor &m
                       , bool isScarce
                       , const NoiseConfig& noiseConfig
                       , PVec<> pos
                       )
   : m_noiseConfig(noiseConfig)
   , m_pos(pos)
{
   setData(m, isScarce);
}


DataConfig::~DataConfig()
{
}

void DataConfig::check() const
{
   THROWERROR_ASSERT(m_dims.size() > 0);

   if (isDense())
   {
       THROWERROR_ASSERT(m_nnz == std::accumulate(m_dims.begin(), m_dims.end(), 1ULL, std::multiplies<std::uint64_t>()));
   }
}

//
// other methods
//

void DataConfig::setData(const Matrix &m)
{
   m_dense_matrix_data = m;
   m_isDense = true;
   m_isScarce = false;
   m_isMatrix = true;
   m_dims = { (std::uint64_t)m.rows(), (std::uint64_t)m.cols() };
   m_nnz = m.nonZeros();
   check();
}

void DataConfig::setData(const SparseMatrix &m, bool isScarce)
{
   m_sparse_matrix_data = m;
   m_isDense = false;
   m_isScarce = isScarce;
   m_isMatrix = true;
   m_dims = { (std::uint64_t)m.rows(), (std::uint64_t)m.cols() };
   m_nnz = m.nonZeros();
   check();
}

void DataConfig::setData(const DenseTensor &m)
{
   m_dense_tensor_data = m;
   m_isDense = true;
   m_isMatrix = false;
   m_dims = m.getDims();
   m_nnz = m.getNNZ();
}

void DataConfig::setData(const SparseTensor &m, bool isScarce)
{
   m_sparse_tensor_data = m;
   m_isDense = false;
   m_isScarce = isScarce;
   m_isMatrix = false;
   m_dims = m.getDims();
   m_nnz = m.getNNZ();
   check();
}

const Matrix &DataConfig::getDenseMatrixData() const
{
   THROWERROR_ASSERT(isDense() && isMatrix());
   return m_dense_matrix_data;
}

const SparseMatrix &DataConfig::getSparseMatrixData() const
{
   THROWERROR_ASSERT(!isDense() && isMatrix());
   return m_sparse_matrix_data;
}

const SparseTensor &DataConfig::getSparseTensorData() const
{
   THROWERROR_ASSERT(!isDense() && !isMatrix());
   return m_sparse_tensor_data;
}

const DenseTensor &DataConfig::getDenseTensorData() const
{
   THROWERROR_ASSERT(isDense() && !isMatrix());
   return m_dense_tensor_data;
}

bool DataConfig::isDense() const
{
   return m_isDense;
}

bool DataConfig::isScarce() const
{
   return m_isScarce;
}

bool DataConfig::isMatrix() const
{
   return m_isMatrix;
}

std::uint64_t DataConfig::getNNZ() const
{
   return m_nnz;
}

std::uint64_t DataConfig::getNModes() const
{
   return m_dims.size();
}

const std::vector<std::uint64_t>& DataConfig::getDims() const
{
   return m_dims;
}

const NoiseConfig& DataConfig::getNoiseConfig() const
{
   return m_noiseConfig;
}

void DataConfig::setNoiseConfig(const NoiseConfig& value)
{
   m_noiseConfig = value;
}

void DataConfig::setFilename(const std::string &f)
{
    m_filename = f;
}

const std::string &DataConfig::getFilename() const
{
    return m_filename;
}

void DataConfig::setPos(const PVec<>& p)
{
   m_pos = p;
}

bool DataConfig::hasPos() const
{
    return m_pos.size();
}

const PVec<>& DataConfig::getPos() const
{
    return m_pos;
}

std::ostream& DataConfig::info(std::ostream& os) const
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

std::string DataConfig::info() const
{
    std::stringstream ss;
    info(ss);
    return ss.str();
}

void DataConfig::save(ConfigFile& writer, const std::string& sectionName) const
{
   //write tensor config position
   if (this->hasPos())
   {
      std::stringstream ss;
      ss << this->getPos();
      writer.put(sectionName, POS_TAG, ss.str());
   }

   //write tensor config filename
   if (isMatrix() && isDense()) 
      writer.write(sectionName, DATA_TAG, getDenseMatrixData());
   else if (isMatrix() && !isDense()) 
      writer.write(sectionName, DATA_TAG, getSparseMatrixData());
   else 
      writer.write(sectionName, DATA_TAG, getSparseTensorData());

   writer.put(sectionName, FILE_TAG, this->getFilename());

   //write tensor config type
   std::string type_str = this->isDense() ? DENSE_TAG : this->isScarce() ? SCARCE_TAG : SPARSE_TAG;
   writer.put(sectionName, TYPE_TAG, type_str);

   //write noise config
   auto &noise_config = this->getNoiseConfig();
   if (noise_config.getNoiseType() != NoiseTypes::unset)
   {
      writer.put(sectionName, NOISE_MODEL_TAG, noiseTypeToString(noise_config.getNoiseType()));
      writer.put(sectionName, PRECISION_TAG, noise_config.getPrecision());
      writer.put(sectionName, SN_INIT_TAG, noise_config.getSnInit());
      writer.put(sectionName, SN_MAX_TAG, noise_config.getSnMax());
      writer.put(sectionName, NOISE_THRESHOLD_TAG, noise_config.getThreshold());
   }

}

bool DataConfig::restore(const ConfigFile& cfg_file, const std::string& sec_name)
{
   //restore position
   std::string pos_str = cfg_file.get(sec_name, POS_TAG, NONE_VALUE);
   if (pos_str != NONE_VALUE)
   {
      std::vector<int> tokens;
      split(pos_str, tokens, ',');

      //assign position
      this->setPos(PVec<>(tokens));
   }

   //restore type
   m_isDense = cfg_file.get(sec_name, TYPE_TAG, DENSE_TAG) == DENSE_TAG;
   m_isScarce = cfg_file.get(sec_name, TYPE_TAG, SCARCE_TAG) == SCARCE_TAG;

   //restore filename and content
   std::string filename = cfg_file.get(sec_name, FILE_TAG, NONE_VALUE);
   if (filename != NONE_VALUE)
   { 
      THROWERROR_NOTIMPL();
      /*
      m_isMatrix = true; // FIXME matrix_io::isMatrixExtension(filename);
      if (isMatrix())
      {
         if (isDense())
            matrix_io::eigen::read_matrix(filename, m_dense_matrix_data);
         else
            matrix_io::eigen::read_matrix(filename, m_sparse_matrix_data);
      }
      else
      {
         THROWERROR_NOTIMPL();
      }
*/
   }
      
   //restore noise model
   NoiseConfig noise;

   NoiseTypes noiseType = stringToNoiseType(cfg_file.get(sec_name, NOISE_MODEL_TAG, noiseTypeToString(NoiseTypes::unset)));
   if (noiseType != NoiseTypes::unset)
   {
      noise.setNoiseType(noiseType);
      noise.setPrecision(cfg_file.get(sec_name, PRECISION_TAG, NoiseConfig::PRECISION_DEFAULT_VALUE));
      noise.setSnInit(cfg_file.get(sec_name, SN_INIT_TAG, NoiseConfig::ADAPTIVE_SN_INIT_DEFAULT_VALUE));
      noise.setSnMax(cfg_file.get(sec_name, SN_MAX_TAG, NoiseConfig::ADAPTIVE_SN_MAX_DEFAULT_VALUE));
      noise.setThreshold(cfg_file.get(sec_name, NOISE_THRESHOLD_TAG, NoiseConfig::PROBIT_DEFAULT_VALUE));
   }

   //assign noise model
   this->setNoiseConfig(noise);

   return true;
}


std::shared_ptr<Data> DataConfig::create(std::shared_ptr<IDataCreator> creator) const
{
   return creator->create(shared_from_this());
}

void DataConfig::write(std::shared_ptr<IDataWriter> writer) const
{
   THROWERROR_NOTIMPL()
}


std::shared_ptr<DataConfig> DataConfig::restore_data_config(const ConfigFile& cfg_file, const std::string& sec_name)
{
   //restore filename
   std::string filename = cfg_file.get(sec_name, FILE_TAG, NONE_VALUE);
   if (filename == NONE_VALUE)
      return std::shared_ptr<DataConfig>();

   //restore type
   bool is_scarce = cfg_file.get(sec_name, TYPE_TAG, SCARCE_TAG) == SCARCE_TAG;

   //restore data
   //auto cfg = generic_io::read_data_config(filename, is_scarce);
   auto cfg = std::make_shared<DataConfig>();

   //restore instance
   cfg->restore(cfg_file, sec_name);

   return cfg;
}

} // end namespace smurff
