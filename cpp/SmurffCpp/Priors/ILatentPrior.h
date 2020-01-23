#pragma once

#include <memory>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>

#include <SmurffCpp/Sessions/Session.h>
#include <SmurffCpp/Noises/INoiseModel.h>
#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/Utils/Distribution.h>
#include <Utils/ThreadVector.hpp>

#include <SmurffCpp/Model.h>

#include <SmurffCpp/VMatrixIterator.hpp>
#include <SmurffCpp/ConstVMatrixIterator.hpp>

namespace smurff {

class StepFile;

class ILatentPrior
{
   
private:
   std::weak_ptr<Session> m_session;

public:
   Session &getSession() const { return *m_session.lock(); }
   std::uint32_t m_mode;
   std::string m_name = "xxxx";

   thread_vector<Vector> rrs;
   thread_vector<Matrix> MMs;

protected:
   ILatentPrior(){}

public:
   ILatentPrior(std::shared_ptr<Session> session, uint32_t mode, std::string name = "xxxx");
   virtual ~ILatentPrior() {}
   virtual void init();

   // utility
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

   virtual bool save(std::shared_ptr<const StepFile> sf) const;
   virtual void restore(std::shared_ptr<const StepFile> sf);
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
