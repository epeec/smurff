#pragma once

#include <memory>

#include <SmurffCpp/Types.h>

#include <SmurffCpp/Utils/Distribution.h>

#include <SmurffCpp/Priors/ILatentPrior.h>

namespace smurff {

// Spike and slab prior
class NormalOnePrior : public ILatentPrior 
{
public:
  // hyperparams
  Vector &mu() { return model().getMu(getMode()); } 
  const Vector &mu() const { return model().getMu(getMode()); } 

  Matrix Lambda;
  Matrix WI;
  Vector mu0;

  // constants
  int b0;
  int df;

public:
   NormalOnePrior(TrainSession &trainSession, uint32_t mode, std::string name = "NormalOnePrior");
   virtual ~NormalOnePrior() {}
   void init() override;

   //mu in NormalPrior does not depend on column index
   //however successors of this class can override this method
   //for example in MacauPrior mu depends on Uhat.col(n)
   virtual const Vector fullMu(int n) const;

   void sample_latent(int n) override;
   virtual std::pair<float_type,float_type> sample_latent(int d, int k, const Matrix& XX, const Vector& yX);

   void update_prior() override;

   // mean value of Z
   std::ostream &status(std::ostream &os, std::string indent) const override;
};
}
