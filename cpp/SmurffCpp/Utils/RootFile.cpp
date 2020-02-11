#include "RootFile.h"

#include <iostream>
#include <fstream>

#include <highfive/H5File.hpp>

#include <Utils/Error.h>
#include <Utils/StringUtils.h>
#include <SmurffCpp/IO/GenericIO.h>
#include <SmurffCpp/StatusItem.h>

#define OPTIONS_TAG "options"
#define STEPS_TAG "steps"
#define STATUS_TAG "status"
#define LAST_CHECKPOINT_TAG "last_checkpoint"
#define LAST_SAMPLE_TAG "last_checkpoint"

#define CHECKPOINT_PREFIX "checkpoint_"
#define SAMPLE_PREFIX "sample_"

namespace h5 = HighFive;

namespace smurff {

RootFile::RootFile(std::string path, bool create)
   : m_path(path)
   , m_h5(path, create ? h5::File::Create : h5::File::ReadOnly)
{
}

std::string RootFile::getFullPath() const
{
   return m_path;
}

std::string RootFile::getPrefix() const
{
   return dirName(m_path);
}

std::string RootFile::getOptionsFileName() const
{
   return getPrefix() + "options.ini";
}

std::string RootFile::getCsvStatusFileName() const
{
   return getPrefix() + "status.csv";
}

void RootFile::createCsvStatusFile()
{
   //write header to status file
   const std::string statusPath = getCsvStatusFileName();
   std::ofstream csv_out(getCsvStatusFileName(), std::ofstream::out);
   csv_out << StatusItem::getCsvHeader() << std::endl;
   m_h5.createAttribute<std::string>(STATUS_TAG, statusPath);
}

void RootFile::addCsvStatusLine(const StatusItem &status_item) const
{
    const std::string statusPath = getCsvStatusFileName();
    std::ofstream csv_out(statusPath, std::ofstream::out | std::ofstream::app);
    THROWERROR_ASSERT_MSG(csv_out, "Could not open status csv file: " + statusPath);;
    csv_out << status_item.asCsvString() << std::endl;
}

void RootFile::saveConfig(Config& config)
{
   std::string configPath = getOptionsFileName();
   config.save(configPath);
   m_h5.createAttribute<std::string>(OPTIONS_TAG, configPath);
}

std::string RootFile::restoreGetOptionsFileName() const
{
   std::string options_filename;
   m_h5.getAttribute(OPTIONS_TAG).read(options_filename);
   return options_filename;
}

void RootFile::restoreConfig(Config& config)
{
   //get options filename
   std::string optionsFileName = restoreGetOptionsFileName();

   //restore config
   bool success = config.restore(optionsFileName);
   THROWERROR_ASSERT_MSG(success, "Could not load ini file '" + optionsFileName + "'");
}

std::shared_ptr<StepFile> RootFile::createSampleStepFile(std::int32_t isample)
{
   return createStepFileInternal(isample, false);
}

std::shared_ptr<StepFile> RootFile::createCheckpointStepFile(std::int32_t isample)
{
   return createStepFileInternal(isample, true);
}

std::shared_ptr<StepFile> RootFile::createStepFileInternal(std::int32_t isample, bool checkpoint)
{
   std::string name = std::string(checkpoint ? CHECKPOINT_PREFIX : SAMPLE_PREFIX) + std::to_string(isample);
   h5::Group group = m_h5.createGroup(name);
   return std::make_shared<StepFile>(group, isample, checkpoint);
}

void RootFile::removeSampleStepFile(std::int32_t isample) 
{
   removeStepFileInternal(isample, false);
}

void RootFile::removeCheckpointStepFile(std::int32_t isample)
{
   removeStepFileInternal(isample, true);
}

void RootFile::removeStepFileInternal(std::int32_t isample, bool checkpoint)
{
   std::string m_name = std::string(checkpoint ? CHECKPOINT_PREFIX : SAMPLE_PREFIX) + std::to_string(isample);
   m_h5.unlink(m_name);
}

std::shared_ptr<StepFile> RootFile::openLastCheckpoint() const
{
   std::string lastCheckpointItem;
   m_h5.getAttribute(LAST_CHECKPOINT_TAG).read(lastCheckpointItem);
   h5::Group group = m_h5.getGroup(lastCheckpointItem);
   return std::make_shared<StepFile>(group);
}

std::vector<std::shared_ptr<StepFile>> RootFile::openSampleStepFiles() const
{
   std::vector<std::string> h5_objects = m_h5.listObjectNames();
   std::vector<std::shared_ptr<StepFile>> samples;

   for (auto& name : h5_objects)
   {
      if (startsWith(name, SAMPLE_PREFIX))
      {
         h5::Group group = m_h5.getGroup(name);
         samples.push_back(std::make_shared<StepFile>(group));
      }
   }

   return samples;
}

std::string RootFile::getFullPathFromIni(const std::string &section, const std::string &field) const
{
   std::string item;
   m_h5.getAttribute(section).read(item);

   if (startsWith(item, getPrefix()))
      return item;

   return getPrefix() + item;
}

} // end namespace smurff
