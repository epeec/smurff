#pragma once

#include <memory>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>

#include <SmurffCpp/Utils/Distribution.h>
#include <Utils/ThreadVector.hpp>

#include <SmurffCpp/Priors/ILatentPrior.h>

namespace smurff {

// Prior without side information (pure BPMF) 
class NormalPrior : public ILatentPrior 
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
  NormalPrior(Session &session, uint32_t mode, std::string name = "NormalPrior");
  virtual ~NormalPrior() {}
  void init() override;

  //mu in NormalPrior does not depend on column index
  //however successors of this class can override this method
  //for example in MacauPrior mu depends on Uhat.col(n)
  virtual const Vector fullMu(int n) const;
  const Matrix getLambda(int n) const;
  
  void sample_latent(int n) override;

  void update_prior() override;
  std::ostream &status(std::ostream &os, std::string indent) const override;
};
}
