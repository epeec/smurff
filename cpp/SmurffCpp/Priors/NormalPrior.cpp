#include "NormalPrior.h"

#include <iomanip>

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <Utils/counters.h>

#include <SmurffCpp/Types.h>

#include <SmurffCpp/Utils/Distribution.h>
#include <Utils/Error.h>

namespace smurff {

//  base class NormalPrior

NormalPrior::NormalPrior(TrainSession &trainSession, uint32_t mode, std::string name)
   : ILatentPrior(trainSession, mode, name)
{}

void NormalPrior::init()
{
   //does not look that there was such init previously
   ILatentPrior::init();

   const int K = num_latent();
   mu().resize(K);
   mu().setZero();

   Lambda.resize(K, K);
   Lambda.setIdentity();
   Lambda *= 10;

   // parameters of Inv-Whishart distribution
   WI.resize(K, K);
   WI.setIdentity();
   mu0.resize(K);
   mu0.setZero();
   b0 = 2;
   df = K;

   const auto &config = getConfig();
   if (config.hasPropagatedPosterior(getMode()))
   {
      m_name += " with posterior propagation";
   }
}

const Vector NormalPrior::fullMu(int n) const
{
   if (getConfig().hasPropagatedPosterior(getMode()))
   {
      return getConfig().getMuPropagatedPosterior(getMode()).getDenseMatrixData().row(n);
   }
   //else
   return mu();
}

const Matrix NormalPrior::getLambda(int n) const
{
   if (getConfig().hasPropagatedPosterior(getMode()))
   {
      const auto &Lambda_pp = getConfig().getLambdaPropagatedPosterior(getMode()).getDenseMatrixData();
      return Eigen::Map<const Matrix>(Lambda_pp.row(n).data(), num_latent(), num_latent());
   }
   //else
   return Lambda;
}
void NormalPrior::update_prior()
{
   SHOW(U());
   SHOW(getUsum());
   SHOW(getUUsum());
   std::tie(mu(), Lambda) = CondNormalWishart(num_item(), getUUsum(), getUsum(), mu0, b0, WI, df);
}

//n is an index of column in U matrix
void  NormalPrior::sample_latent(int n)
{
   const auto &mu_u = fullMu(n);
   const auto &Lambda_u = getLambda(n);

   Vector &rr = rrs.local();
   Matrix &MM = MMs.local();

   rr.setZero();
SHOW(rr);
   MM.setZero();

   // add pnm
   data().getMuLambda(model(), m_mode, n, rr, MM);
SHOW(rr);

   // add hyperparams
   rr.noalias() += mu_u * Lambda_u;
   SHOW(mu_u);
   SHOW(Lambda_u);
   SHOW(mu_u * Lambda_u);
   SHOW(rr);
   MM.noalias() += Lambda_u;

   //Solve system of linear equations for x: MM * x = rr - not exactly correct  because we have random part
   //Sample from multivariate normal distribution with mean rr and precision matrix MM

   Eigen::LLT<Matrix> chol;
   {
      chol = MM.llt(); // compute the Cholesky decomposition X = L * U
      if(chol.info() != Eigen::Success)
      {
         THROWERROR("Cholesky Decomposition failed!");
      }
   }

   chol.matrixL().solveInPlace(rr.transpose()); // solve for y: y = L^-1 * b
SHOW(rr);
   rr.noalias() += nrandn(num_latent());
SHOW(rr);
   chol.matrixU().solveInPlace(rr.transpose()); // solve for x: x = U^-1 * y
SHOW(rr);
   
   U().row(n).noalias() = rr; // rr is equal to x
}

std::ostream &NormalPrior::status(std::ostream &os, std::string indent) const
{
   os << indent << m_name << std::endl;
   os << indent << "  mu: " <<  mu() << std::endl;
   return os;
}
} // end namespace smurff
