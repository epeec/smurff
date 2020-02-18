#include <iostream>
#include <fstream>
#include <memory>

#include <SmurffCpp/IO/INIFile.h>
#include <SmurffCpp/Utils/HDF5.h>

#include <Utils/StringUtils.h>

#include "SideInfoConfig.h"
#include "TensorConfig.h"


#define SIDE_INFO_PREFIX "side_info"
#define TOL_TAG "tol"
#define DIRECT_TAG "direct"
#define THROW_ON_CHOLESKY_ERROR_TAG "throw_on_cholesky_error"
#define NUMBER_TAG "nr"

namespace smurff {

double SideInfoConfig::BETA_PRECISION_DEFAULT_VALUE = 10.0;
double SideInfoConfig::TOL_DEFAULT_VALUE = 1e-6;

SideInfoConfig::SideInfoConfig()
{
   m_tol = SideInfoConfig::TOL_DEFAULT_VALUE;
   m_direct = false;
   m_throw_on_cholesky_error = false;
}

template<class ConfigFile>
void SideInfoConfig::save(ConfigFile& cfg_file, std::size_t prior_index) const
{
   std::string sectionName = addIndex(SIDE_INFO_PREFIX, prior_index);

   //macau data
   cfg_file.put(sectionName, TOL_TAG, m_tol);
   cfg_file.put(sectionName, DIRECT_TAG, m_direct);
   cfg_file.put(sectionName, THROW_ON_CHOLESKY_ERROR_TAG, m_throw_on_cholesky_error);

   //TensorConfig data
   TensorConfig::save_tensor_config(cfg_file, sectionName, -1, m_sideInfo);
}

template
void SideInfoConfig::save(INIFile& cfg_file, std::size_t prior_index) const;

template
void SideInfoConfig::save(HDF5& cfg_file, std::size_t prior_index) const;

template<class ConfigFile>
bool SideInfoConfig::restore(const ConfigFile& cfg_file, std::size_t prior_index)
{
   std::string sectionName = addIndex(SIDE_INFO_PREFIX, prior_index);

   if (!cfg_file.hasSection(sectionName))
   {
       return false;
   }

   //restore side info properties
   m_tol = cfg_file. template get<double>(sectionName, TOL_TAG, SideInfoConfig::TOL_DEFAULT_VALUE);
   m_direct = cfg_file. template get<bool>(sectionName, DIRECT_TAG, false);
   m_throw_on_cholesky_error = cfg_file. template get<bool>(sectionName, THROW_ON_CHOLESKY_ERROR_TAG, false);

   auto tensor_cfg = TensorConfig::restore_tensor_config(cfg_file, sectionName);
   m_sideInfo = std::dynamic_pointer_cast<MatrixConfig>(tensor_cfg);

   return (bool)m_sideInfo;
}

template
bool SideInfoConfig::restore(const INIFile& cfg_file, std::size_t prior_index);

template
bool SideInfoConfig::restore(const HDF5& cfg_file, std::size_t prior_index);

} // end namespace smurff
