#include "OutputFile.h"

#include <iostream>
#include <fstream>

#include <highfive/H5File.hpp>

#include <Utils/Error.h>
#include <Utils/StringUtils.h>
#include <SmurffCpp/IO/INIFile.h>
#include <SmurffCpp/IO/GenericIO.h>
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

OutputFile::OutputFile(std::string path, bool create)
   : m_path(path)
   , m_h5(path, create ? h5::File::Create : h5::File::ReadWrite)
{
   if (create)
   {
      m_h5.createAttribute(LAST_CHECKPOINT_TAG, std::string(NONE_VALUE));
   }
}

std::string OutputFile::getFullPath() const
{
   return m_path;
}

std::string OutputFile::getPrefix() const
{
   return dirName(m_path);
}

std::string OutputFile::getOptionsFileName() const
{
   return getPrefix() + "options.ini";
}

void OutputFile::saveConfig(Config& config)
{
   // save to INIFile
   INIFile cfg_file;
   config.save(cfg_file);
   cfg_file.write(getOptionsFileName());

   // save to HDF5
   HDF5 h5_cfg(m_h5.createGroup(OPTIONS_TAG));
   config.save(h5_cfg);
}

std::string OutputFile::restoreGetOptionsFileName() const
{
   std::string options_filename;
   m_h5.getAttribute(OPTIONS_TAG).read(options_filename);
   return options_filename;
}

void OutputFile::restoreConfig(Config& config)
{
   //get options filename
   std::string optionsFileName = restoreGetOptionsFileName();
   THROWERROR_FILE_NOT_EXIST(optionsFileName);

   //restore config
   INIFile cfg_file;
   cfg_file.read(optionsFileName);

   bool success = config.restore(cfg_file);
   THROWERROR_ASSERT_MSG(success, "Could not load ini file '" + optionsFileName + "'");
}

std::shared_ptr<Step> OutputFile::createSampleStep(std::int32_t isample)
{
   return createStep(isample, false);
}

std::shared_ptr<Step> OutputFile::createCheckpointStep(std::int32_t isample)
{
   return createStep(isample, true);
}

std::shared_ptr<Step> OutputFile::createStep(std::int32_t isample, bool checkpoint)
{
   return std::make_shared<Step>(m_h5, isample, checkpoint);
}

void OutputFile::removeOldCheckpoints()
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

std::shared_ptr<Step> OutputFile::openLastCheckpoint() const
{
   std::string lastCheckpointItem;
   m_h5.getAttribute(LAST_CHECKPOINT_TAG).read(lastCheckpointItem);
   if (lastCheckpointItem != NONE_VALUE)
   {
      h5::Group group = m_h5.getGroup(lastCheckpointItem);
      return std::make_shared<Step>(m_h5, group);
   }

   return std::shared_ptr<Step>();
}

std::vector<std::shared_ptr<Step>> OutputFile::openSampleSteps() const
{
   std::vector<std::string> h5_objects = m_h5.listObjectNames();
   std::vector<std::shared_ptr<Step>> samples;

   for (auto &name : h5_objects)
   {
      if (startsWith(name, SAMPLE_PREFIX))
      {
         h5::Group group = m_h5.getGroup(name);
         samples.push_back(std::make_shared<Step>(m_h5, group));
      }
   }

   return samples;
}

} // end namespace smurff
