#pragma once

#include <SmurffCpp/Sessions/TrainSession.h>

namespace smurff {

class PythonSession : public TrainSession
{

private:
   static bool keepRunning;
   static bool keepRunningVerbose;

public:
   PythonSession()
   {
      name = "PythonSession";
      keepRunning = true;
   }
  
   void setPriorTypes(std::vector<std::string> values) { m_config.setPriorTypes(values); }   
   void setRestoreName(std::string value) { m_config.setRestoreName(value); } 
   void setSaveName(std::string value) { m_config.setSaveName(value); } 
   void setSaveFreq(int value) { m_config.setSaveFreq(value); } 
   void setSavePred(bool value) { m_config.setSavePred(value); } 
   void setCheckpointFreq(int value) { m_config.setCheckpointFreq(value); } 
   void setRandomSeed(int value) { m_config.setRandomSeed(value); } 
   void setVerbose(int value) { m_config.setVerbose(value); } 
   void setBurnin(int value) { m_config.setBurnin(value); } 
   void setNSamples(int value) { m_config.setNSamples(value); } 
   void setNumLatent(int value) { m_config.setNumLatent(value); } 
   void setThreshold(double value) { m_config.setThreshold(value); } 
   void setNumThreads(int value) { m_config.setNumThreads(value); }

   template <typename SparseType>
   void setTest(const SparseType &data)
   {
       m_config.getTest().setData(data, true);
   }

   void addSideInfoDense(int mode, const Matrix &data, const NoiseConfig &nc, bool direct) 
   {
      auto &si = m_config.addSideInfo(mode);
      si.setData(data);
      si.setNoiseConfig(nc);
      si.setDirect(direct);
   }

   void addSideInfoSparse(int mode, const SparseMatrix &data, const NoiseConfig &nc, bool direct) 
   {
      auto &si = m_config.addSideInfo(mode);
      si.setData(data, false);
      si.setNoiseConfig(nc);
      si.setDirect(direct);
   }

   template <typename DenseType>
   void addDataDense(std::vector<int> pos, const DenseType &data, const NoiseConfig &nc)
   {
      auto &data_config = m_config.addData();
      data_config.setPos(pos);
      data_config.setData(data);
      data_config.setNoiseConfig(nc);
   }
 
   template <typename SparseType>
   void addDataSparse(std::vector<int> pos, const SparseType &data, const NoiseConfig &nc, bool is_scarce)
   {
      auto &data_config = m_config.addData();
      data_config.setPos(pos);
      data_config.setData(data, is_scarce);
      data_config.setNoiseConfig(nc);
   }

   void addPropagatedPosterior(int mode, const Matrix &mu, const Matrix &Lambda)
   {
      m_config.addPropagatedPosterior(mode, mu, Lambda);
   }

   bool interrupted() override
   {
       return !keepRunning;
   }

   bool step() override;

private:
   static void intHandler(int);
};

}
