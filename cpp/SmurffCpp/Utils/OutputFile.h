#pragma once

#include <string>
#include <unordered_map>

#include <highfive/H5File.hpp>

namespace h5 = HighFive;

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Utils/Step.h>

namespace smurff {

struct StatusItem;


extern const std::string LAST_CHECKPOINT_TAG;

class OutputFile
{
private:
   std::string m_path;
   h5::File m_h5;

public:
   OutputFile(std::string path, bool create = false);

public:
   std::string getPrefix() const;
   std::string getFullPath() const;
   std::string getOptionsFileName() const;

public:
   void saveConfig(Config& config);
   void restoreConfig(Config& config);

private:
   std::string restoreGetOptionsFileName() const;

public:
   std::shared_ptr<Step> createSampleStep(std::int32_t isample);
   std::shared_ptr<Step> createCheckpointStep(std::int32_t isample);
   std::shared_ptr<Step> createStep(std::int32_t isample, bool checkpoint);

public:
   void removeOldCheckpoints();

public:
   std::shared_ptr<Step> openLastCheckpoint() const;
   std::shared_ptr<Step> openSampleStep(int isample) const;
   std::vector<std::shared_ptr<Step>> openSampleSteps() const;
};

}
