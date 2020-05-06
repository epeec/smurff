#pragma once

#include <string>

#include <highfive/H5File.hpp>

namespace h5 = HighFive;

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Utils/SaveState.h>

namespace smurff {

struct StatusItem;

extern const std::string LAST_CHECKPOINT_TAG;

class StateFile
{
private:
   std::string m_path;
   h5::File m_h5;

public:
   StateFile(std::string path, bool create = false);

public:
   std::string getPath() const;

public:
   void saveConfig(const Config& config);
   void restoreConfig(Config& config);

public:
   SaveState createSampleStep(std::int32_t isample, bool save_aggr);
   SaveState createStep(std::int32_t isample, bool checkpoint, bool save_aggr);

public:
   void removeOldCheckpoints();

public:
   bool hasCheckpoint() const;
   SaveState openCheckpoint() const;
   SaveState openSampleStep(int isample) const;
   std::vector<SaveState> openSampleSteps() const;
};

}
