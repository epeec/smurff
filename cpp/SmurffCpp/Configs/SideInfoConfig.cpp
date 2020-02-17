#include <iostream>
#include <fstream>
#include <memory>

#include "SideInfoConfig.h"
#include "TensorConfig.h"

#include <SmurffCpp/IO/INIFile.h>

#define MACAU_PRIOR_CONFIG_PREFIX_TAG "macau_prior_config"
#define MACAU_PRIOR_CONFIG_ITEM_PREFIX_TAG "macau_prior_config_item"

#define NUM_SIDE_INFO_TAG "num_side_info"

#define TOL_TAG "tol"
#define DIRECT_TAG "direct"
#define THROW_ON_CHOLESKY_ERROR_TAG "throw_on_cholesky_error"
#define SIDE_INFO_PREFIX "side_info"
#define MODE_TAG "mode"
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

void SideInfoConfig::save(INIFile& writer, std::size_t prior_index) const
{
   std::string sectionName = INIFile::add_index(MACAU_PRIOR_CONFIG_ITEM_PREFIX_TAG, prior_index);

   //macau data
   writer.put(sectionName, TOL_TAG, m_tol);
   writer.put(sectionName, DIRECT_TAG, m_direct);
   writer.put(sectionName, THROW_ON_CHOLESKY_ERROR_TAG, m_throw_on_cholesky_error);
   writer.put(sectionName, MODE_TAG, prior_index);

   //TensorConfig data
   TensorConfig::save_tensor_config(writer, sectionName, -1, m_sideInfo);
}

bool SideInfoConfig::restore(const INIFile& reader, std::size_t prior_index)
{
   std::string sectionName = INIFile::add_index(MACAU_PRIOR_CONFIG_ITEM_PREFIX_TAG, prior_index);

   if (!reader.hasSection(sectionName))
   {
       return false;
   }

   //restore side info properties
   m_tol = reader.get<double>(sectionName, TOL_TAG, SideInfoConfig::TOL_DEFAULT_VALUE);
   m_direct = reader.get<bool>(sectionName, DIRECT_TAG, false);
   m_throw_on_cholesky_error = reader.get<bool>(sectionName, THROW_ON_CHOLESKY_ERROR_TAG, false);

   int mode = reader.get<int>(sectionName, MODE_TAG, -1);
   THROWERROR_ASSERT(mode == prior_index);

   auto tensor_cfg = TensorConfig::restore_tensor_config(reader, sectionName);
   m_sideInfo = std::dynamic_pointer_cast<MatrixConfig>(tensor_cfg);

   return true;
}
} // end namespace smurff
