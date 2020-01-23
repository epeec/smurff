#pragma once

#include <memory>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>

#include <SmurffCpp/Utils/Distribution.h>
#include <Utils/ThreadVector.hpp>

#include <SmurffCpp/Priors/NormalOnePrior.h>

namespace smurff {

// Spike and slab prior
class SpikeAndSlabPrior : public NormalOnePrior 
{
public:
   // updated by every thread during sample_latents
   thread_vector<Matrix> Zcol, W2col;

   // updated during update_prior
   Array2D Zkeep;
   Array2D alpha, log_alpha;
   Array2D r, log_r;

   //-- hyper params
   const double prior_beta = 1; //for r
   const double prior_alpha_0 = 1.; //for alpha
   const double prior_beta_0 = 1.; //for alpha

public:
   SpikeAndSlabPrior(std::shared_ptr<Session> session, uint32_t mode);
   virtual ~SpikeAndSlabPrior() {}
   void init() override;

   void restore(std::shared_ptr<const StepFile> sf) override;

   std::pair<double,double> sample_latent(int d, int k, const Matrix& XX, const Vector& yX) override;

   void update_prior() override;

   // mean value of Z
   std::ostream &status(std::ostream &os, std::string indent) const override;
};
}
