#pragma once

#include <iostream>
#include <memory>

#include <Utils/Error.h>
#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Priors/IPriorFactory.h>
#include <SmurffCpp/Utils/StateFile.h>
#include <SmurffCpp/StatusItem.h>
#include <SmurffCpp/Sessions/ISession.h>
#include <SmurffCpp/Model.h>
#include <SmurffCpp/result.h>

namespace smurff {


class TrainSession : public ISession
{
public:
   TrainSession() : name("TrainSession") { };
   TrainSession(const Config &c) : ISession(c), name("TrainSession") {}

protected:
   Model m_model;
   Result m_pred;

protected:
   std::vector<std::shared_ptr<ILatentPrior> > m_priors;
   std::string name;

   //train data
   std::shared_ptr<Data> data_ptr;

private:
   std::shared_ptr<StateFile> m_stateFile;

private:
   int m_iter = -1; //index of step iteration
   double m_secs_per_iter = .0; //time in seconds for last_iter
   double m_secs_total = .0; //time in seconds for last_iter
   double m_lastCheckpointTime;
   int m_lastCheckpointIter;

public:
   bool inBurninPhase() const { return m_iter < getConfig().getBurnin(); }
   bool inSamplingPhase() const { return !inBurninPhase(); }
   bool finalSample() const { return m_iter == (getConfig().getNSamples() + getConfig().getBurnin()); }

public:
   const Result &getResult() const override;

   // execution of the sampler
public:
   void run() override;
   void init() override;
   bool step() override;

public:
   std::ostream &info(std::ostream &, std::string indent) const override;

private:
   //save current iteration
   void save();

   void saveInternal(int iteration, bool checkpoint);

   //restore last iteration
   bool restore(int& iteration);

private:
   void printStatus(std::ostream& output, bool resume = false);

public:
   StatusItem getStatus() const override;

private:
   void initRng();

public:
   virtual std::shared_ptr<IPriorFactory> create_prior_factory() const;

public:
   Data &data() const
   {
      THROWERROR_ASSERT(data_ptr != 0);
      return *data_ptr;
   }

   const Model& model() const
   {
      return m_model;
   }

   Model& model()
   {
      return m_model;
   }


};

} // end namespace
