#pragma once

#include <memory>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>

#include <SmurffCpp/Sessions/TrainSession.h>
#include <SmurffCpp/Noises/INoiseModel.h>
#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/Utils/Distribution.h>
#include <Utils/ThreadVector.hpp>

#include <SmurffCpp/Model.h>

#include <SmurffCpp/VMatrixIterator.hpp>
#include <SmurffCpp/ConstVMatrixIterator.hpp>

namespace smurff {

class SaveState;

class ILatentPrior
{
public:
   TrainSession &m_session;
   std::uint32_t m_mode;
   std::string m_name = "xxxx";

   thread_vector<Vector> rrs;
   thread_vector<Matrix> MMs;

public:
   ILatentPrior(TrainSession &trainSession, uint32_t mode, std::string name = "xxxx");
   virtual ~ILatentPrior() {}
   virtual void init();

   // utility
   const Config &getConfig() const;

   const Model& model() const;
   Model& model();

   Data& data() const;
   double predict(const PVec<> &) const;

   INoiseModel& noise();

   Matrix &U();
   const Matrix &U() const;

   //return V matrices in the model opposite to mode
   VMatrixIterator<Matrix> Vbegin();
   
   VMatrixIterator<Matrix> Vend();

   ConstVMatrixIterator<Matrix> CVbegin() const;
   
   ConstVMatrixIterator<Matrix> CVend() const;

   int num_latent() const;
   int num_item() const;

   const Vector& getUsum() { return Usum; } 
   const Matrix& getUUsum()  { return UUsum; }

   virtual bool save(SaveState &sf) const;
   virtual void restore(const SaveState &sf);
   virtual std::ostream &info(std::ostream &os, std::string indent);
   virtual std::ostream &status(std::ostream &os, std::string indent) const = 0;

   // work
   virtual bool run_slave(); // returns true if some work happened...

   virtual void sample_latents();
   virtual void sample_latent(int n) = 0;

   virtual void update_prior() = 0;

private:
   void init_Usum();
   Vector Usum;
   Matrix UUsum;

   // for effiency, we keep + update Urow and UUrow by every thread
   thread_vector<Vector> Urow;
   thread_vector<Matrix> UUrow;
   
public:
   void setMode(std::uint32_t value)
   {
      m_mode = value;
   }

public:
   std::uint32_t getMode() const
   {
      return m_mode;
   }
};
}
