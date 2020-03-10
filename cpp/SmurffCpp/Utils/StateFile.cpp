#include "StateFile.h"

#include <iostream>
#include <fstream>

#include <highfive/H5File.hpp>

#include <Utils/Error.h>
#include <Utils/StringUtils.h>
#include <SmurffCpp/IO/INIFile.h>
#include <SmurffCpp/StatusItem.h>

namespace h5 = HighFive;

namespace smurff {

const std::string NONE_VALUE = "none";
const std::string OPTIONS_TAG = "options";
const std::string STEPS_TAG = "steps";
const std::string STATUS_TAG = "status";
const std::string LAST_CHECKPOINT_TAG = "last_checkpoint";
const std::string CHECKPOINT_PREFIX = "checkpoint_";
const std::string SAMPLE_PREFIX = "sample_";

StateFile::StateFile(std::string path, bool create)
   : m_path(path)
   , m_h5(path, create ? h5::File::Create : h5::File::ReadWrite)
{
   if (create)
   {
      m_h5.createAttribute(LAST_CHECKPOINT_TAG, std::string(NONE_VALUE));
   }
}

std::string StateFile::getFullPath() const
{
   return m_path;
}

std::string StateFile::getPrefix() const
{
   return dirName(m_path);
}

std::string StateFile::getOptionsFileName() const
{
   return getPrefix() + "options.ini";
}

void StateFile::saveConfig(Config& config)
{
   // save to INIFile
   INIFile cfg_file;
   config.save(cfg_file);
   cfg_file.write(getOptionsFileName());

   // save to HDF5
   HDF5 h5_cfg(m_h5.createGroup(OPTIONS_TAG));
   config.save(h5_cfg);
}

void StateFile::restoreConfig(Config& config)
{
   //get options filename
   HDF5 h5_cfg(m_h5.getGroup(OPTIONS_TAG));
   config.restore(h5_cfg);
}

SaveState StateFile::createSampleStep(std::int32_t isample)
{
   return createStep(isample, false);
}

SaveState StateFile::createCheckpointStep(std::int32_t isample)
{
   return createStep(isample, true);
}

SaveState StateFile::createStep(std::int32_t isample, bool checkpoint)
{
   return SaveState(m_h5, isample, checkpoint);
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
