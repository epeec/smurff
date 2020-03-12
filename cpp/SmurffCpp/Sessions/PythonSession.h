#pragma once

#include <SmurffCpp/Sessions/TrainSession.h>

namespace smurff {

class PythonSession : public TrainSession
{

private:
   static bool keepRunning;
   static bool keepRunningVerbose;

public:
   PythonSession(const Config &c)
   : TrainSession(c)
   {
      name = "PythonSession";
      keepRunning = true;
   }
   
   template <typename DenseType>
   void setTrainDense(const DenseType &data, const NoiseConfig &nc)
   {
      m_config.getTrain().setData(data);
      m_config.getTrain().setNoiseConfig(nc);
   }
 
   template <typename SparseType>
   void setTrainSparse(const SparseType &data, const NoiseConfig &nc, bool is_scarce)
   {
      m_config.getTrain().setData(data, is_scarce);
      m_config.getTrain().setNoiseConfig(nc);
   }

   template <typename SparseType>
   void setTest(const SparseType &data)
   {
       m_config.getTest().setData(data, true);
   }

   void addSideInfoDense(int mode, const Matrix &data, const NoiseConfig &nc, bool direct, double tol) 
   {
      auto &si = m_config.addSideInfo(mode);
      si.setData(data);
      si.setDirect(direct);
      si.setTol(tol);
   }

   void addSideInfoSparse(int mode, const SparseMatrix &data, const NoiseConfig &nc, bool direct, double tol) 
   {
      auto &si = m_config.addSideInfo(mode);
      si.setData(data, false);
      si.setDirect(direct);
      si.setTol(tol);
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
