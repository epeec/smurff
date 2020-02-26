#pragma once

#include <memory>

#include <SmurffCpp/Types.h>

#include <SmurffCpp/Priors/NormalOnePrior.h>

#include <SmurffCpp/SideInfo/ISideInfo.h>

namespace smurff {

//Why remove init method and put everything in constructor if we have
//init method in other priors and the other method addSideInfo which we use in pair

class MacauOnePrior : public NormalOnePrior
{
public:
   Matrix Uhat;

   Vector F_colsq;   // sum-of-squares for every feature (column)

   Matrix &beta() { return model().getLinkMatrix(getMode()); }
   const Matrix &beta() const { return model().getLinkMatrix(getMode()); }
   
   double beta_precision_a0; // Hyper-prior for beta_precision
   double beta_precision_b0; // Hyper-prior for beta_precision

   std::shared_ptr<ISideInfo> Features;  // side information
   Vector beta_precision;
   double bp0;
   bool enable_beta_precision_sampling;

public:
   MacauOnePrior(Session &session, uint32_t mode);

   void init() override;

   void update_prior() override;
    
   const Vector fullMu(int n) const override;

public:
   //FIXME: tolerance_a and direct_a are not really used. 
   //should remove later after PriorFactory is properly implemented. 
   //No reason generalizing addSideInfo between priors
   void addSideInfo(const std::shared_ptr<ISideInfo>& side, double bp, double tol, bool direct, bool enable_beta_precision_sampling, bool throw_on_cholesky_error);

public:

   //used in update_prior

   void sample_beta(const Matrix &U);

   //used in update_prior

   void sample_mu_lambda(const Matrix &U);

   //used in update_prior

   void sample_beta_precision();

public:
   std::ostream& status(std::ostream &os, std::string indent) const override;
};

}
