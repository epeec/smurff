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
  std::shared_ptr<Vector> m_mu; 
  Vector &hyperMu() const { return *m_mu; }

  Matrix Lambda;

  // PP hyperparams
  std::shared_ptr<Matrix> mu_pp; // array of size N to vector of size K 
  std::shared_ptr<Matrix> Lambda_pp; // array of size N  to matrix of size K x K

  Matrix WI;
  Vector mu0;

  // constants
  int b0;
  int df;

protected:
   NormalPrior()
      : ILatentPrior(){}

public:
  NormalPrior(std::shared_ptr<Session> session, uint32_t mode, std::string name = "NormalPrior");
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
