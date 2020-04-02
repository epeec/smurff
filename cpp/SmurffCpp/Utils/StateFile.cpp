#include "StateFile.h"

#include <iostream>
#include <fstream>

#include <highfive/H5File.hpp>

#include <Utils/Error.h>
#include <Utils/StringUtils.h>
#include <SmurffCpp/StatusItem.h>

namespace h5 = HighFive;

namespace smurff {

const std::string NONE_VALUE = "none";
const std::string CONFIG_TAG = "config";
const std::string STEPS_TAG = "steps";
const std::string STATUS_TAG = "status";
const std::string LAST_CHECKPOINT_TAG = "last_checkpoint";
const std::string CHECKPOINT_PREFIX = "checkpoint_";
const std::string SAMPLE_PREFIX = "sample_";

StateFile::StateFile(std::string path, bool create)
   : m_path(path)
   , m_h5(path, create ? h5::File::Overwrite : h5::File::ReadWrite)
{
   if (create)
   {
      m_h5.createAttribute(LAST_CHECKPOINT_TAG, std::string(NONE_VALUE));
   }
}

std::string StateFile::getPath() const
{
   return m_path;
}

void StateFile::saveConfig(const Config& config)
{
   HDF5Group h5_cfg(m_h5.createGroup(CONFIG_TAG));
   config.save(h5_cfg);
}

void StateFile::restoreConfig(Config& config)
{
   HDF5Group h5_cfg(m_h5.getGroup(CONFIG_TAG));
   config.restore(h5_cfg);
}

SaveState StateFile::createSampleStep(std::int32_t isample, bool save_aggr)
{
   return createStep(isample, false, save_aggr);
}

SaveState StateFile::createStep(std::int32_t isample, bool checkpoint, bool save_aggr)
{
   return SaveState(m_h5, isample, checkpoint, save_aggr);
}

void StateFile::removeOldCheckpoints()
{
   std::string lastCheckpointItem;
   m_h5.getAttribute(LAST_CHECKPOINT_TAG).read(lastCheckpointItem);
   //std::cout << "Last checkpoint " << lastCheckpointItem << std::endl;

   std::vector<std::string> h5_objects = m_h5.listObjectNames();
   for (auto &name : h5_objects)
   {
      if (startsWith(name, CHECKPOINT_PREFIX) && name != lastCheckpointItem)
      {
         m_h5.unlink(name);
         //std::cout << "Remove checkpoint " << name << std::endl;
      }
   }
}

bool StateFile::hasCheckpoint() const
{
   std::string lastCheckpointItem;
   m_h5.getAttribute(LAST_CHECKPOINT_TAG).read(lastCheckpointItem);
   return (lastCheckpointItem != NONE_VALUE);
}

SaveState StateFile::openCheckpoint() const
{
   THROWERROR_ASSERT(hasCheckpoint());
   std::string lastCheckpointItem;
   m_h5.getAttribute(LAST_CHECKPOINT_TAG).read(lastCheckpointItem);
   h5::Group group = m_h5.getGroup(lastCheckpointItem);
   return SaveState(m_h5, group);
}

std::vector<SaveState> StateFile::openSampleSteps() const
{
   std::vector<std::string> h5_objects = m_h5.listObjectNames();
   std::vector<SaveState> samples;

   for (auto &name : h5_objects)
   {
      if (startsWith(name, SAMPLE_PREFIX))
      {
         h5::Group group = m_h5.getGroup(name);
         samples.push_back(SaveState(m_h5, group));
      }
   }

   return samples;
}

} // end namespace smurff
