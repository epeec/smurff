#pragma once

#include <string>
#include <unordered_map>

#include <highfive/H5File.hpp>

namespace h5 = HighFive;

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Utils/StepFile.h>

namespace smurff {

struct StatusItem;

class RootFile
{
private:
   std::string m_path;
   h5::File m_h5;

public:
   RootFile(std::string path);


public:
   std::string getPrefix() const;
   std::string getFullPath() const;
   std::string getOptionsFileName() const;
   std::string getCsvStatusFileName() const;

private:
   std::string getFullPathFromIni(const std::string &section, const std::string &field) const;

private:
   void appendToRootFile(std::string name, std::string value) const;

public:
   void saveConfig(Config& config);
   void restoreConfig(Config& config);

private:
   std::string restoreGetOptionsFileName() const;

public:
   std::shared_ptr<StepFile> createSampleStepFile(std::int32_t isample);
   std::shared_ptr<StepFile> createCheckpointStepFile(std::int32_t isample);

public:
   void removeSampleStepFile(std::int32_t isample);
   void removeCheckpointStepFile(std::int32_t isample);

private:
   std::shared_ptr<StepFile> createStepFileInternal(std::int32_t isample, bool burnin);

private:
   void removeStepFileInternal(std::int32_t isample, bool burnin);

public:
   std::shared_ptr<StepFile> openLastCheckpoint() const;
   std::shared_ptr<StepFile> openSampleStepFile(int isample) const;
   std::vector<std::shared_ptr<StepFile>> openSampleStepFiles() const;

public:
  void createCsvStatusFile();
  void addCsvStatusLine(const StatusItem &status_item) const;
};

}
