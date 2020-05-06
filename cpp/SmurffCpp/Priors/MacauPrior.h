#pragma once

#include <memory>


#include <SmurffCpp/Types.h>

#include <SmurffCpp/Priors/NormalPrior.h>
#include <SmurffCpp/SideInfo/ISideInfo.h>

namespace smurff {

//sample_beta method is now virtual. because we override it in MPIMacauPrior
//we also have this method in MacauOnePrior but it is not virtual
//maybe make it virtual?

/// Prior with side information
class MacauPrior : public NormalPrior
{
public:
   Matrix &beta() { return model().getLinkMatrix(getMode()); }
   const Matrix &beta() const { return model().getLinkMatrix(getMode()); }
                   // num_latent x num_feat -- link matrix
   Matrix Uhat;    // num_latent x num_items
   Matrix FtF;     // num_feat   x num feat
   Matrix HyperU;  // num_latent x num_items
   Matrix HyperU2; // num_latent x num_feat
   Matrix Ft_y;    // num_latent x num_feat -- RHS
   Matrix BtB;     // num_latent x num_latent

#ifdef USE_ARRAYFIRE
   // gpu versions
   af::array gpu_FtF;    // num_feat x num_feat
#endif

   int blockcg_iter;
   
   double beta_precision_mu0; // Hyper-prior for beta_precision
   double beta_precision_nu0; // Hyper-prior for beta_precision

   std::shared_ptr<ISideInfo> Features;  // side information
   float_type beta_precision;
   double tol = 1e-6;
   bool use_FtF;
   bool enable_beta_precision_sampling;
   bool throw_on_cholesky_error;
   bool use_ViennaCL = true;

public:
   MacauPrior(TrainSession &trainSession, uint32_t mode);

   virtual ~MacauPrior();

   void init() override;

   void update_prior() override;

   const Vector fullMu(int n) const override;

   int num_feat() const { return Features->cols(); }

   void compute_Ft_y(Matrix& Ft_y);
   virtual void sample_beta();

public:
   void addSideInfo(const std::shared_ptr<ISideInfo>& side_info_a, double beta_precision_a, double tolerance_a, bool direct_a, bool enable_beta_precision_sampling_a, bool throw_on_cholesky_error_a);

public:
   std::ostream& info(std::ostream &os, std::string indent) override;
   std::ostream& status(std::ostream &os, std::string indent) const override;

public:
   static std::pair<double, double> posterior_beta_precision(const Matrix & BtB, Matrix & Lambda_u, double nu, double mu, int N);
   static double sample_beta_precision(const Matrix & BtB, Matrix & Lambda_u, double nu, double mu, int N);
};

}
