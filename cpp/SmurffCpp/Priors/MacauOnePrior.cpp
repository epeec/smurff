#include "MacauOnePrior.h"

namespace smurff {

MacauOnePrior::MacauOnePrior(TrainSession &trainSession, uint32_t mode)
   : NormalOnePrior(trainSession, mode, "MacauOnePrior")
{
   bp0 = SideInfoConfig::BETA_PRECISION_DEFAULT_VALUE;
   enable_beta_precision_sampling = Config::ENABLE_BETA_PRECISION_SAMPLING_DEFAULT_VALUE;
}

void MacauOnePrior::init()
{
   NormalOnePrior::init();

   // init SideInfo related
   Uhat = Matrix::Constant(num_item(), num_latent(), 0.0);
   beta() = Matrix::Constant(num_feat(), num_latent(), 0.0);

   // initial value (should be determined automatically)
   // Hyper-prior for beta_precision (mean 1.0):
   beta_precision = Vector::Constant(num_latent(), bp0);
   beta_precision_a0 = 0.1;
   beta_precision_b0 = 0.1;
}

void MacauOnePrior::update_prior()
{
   sample_mu_lambda(U());
   sample_beta(U());
   Features->compute_uhat(Uhat, beta());

   if (enable_beta_precision_sampling)
      sample_beta_precision();
}

const Vector MacauOnePrior::fullMu(int n) const
{
   return mu() + Uhat.row(n);
}

void MacauOnePrior::addSideInfo(const std::shared_ptr<ISideInfo>& si, double bp, double tol, bool, bool ebps, bool toce)
{
   Features = si;
   bp0 = bp;
   enable_beta_precision_sampling = ebps;
   F_colsq = Features->col_square_sum();
}

void MacauOnePrior::sample_beta(const Matrix &U)
{
   // updating beta() and beta_var
   const int nfeat = num_feat();
   const int nitem = num_item();
   const int blocksize = 4;

   Matrix Z;

   #pragma omp parallel for private(Z) schedule(static, 1)
   for (int dstart = 0; dstart < num_latent(); dstart += blocksize)
   {
      const int dcount = std::min(blocksize, num_latent() - dstart);
      Z.resize(nitem, dcount);

      for (int i = 0; i < nitem; i++)
      {
         for (int d = 0; d < dcount; d++)
         {
            int dx = d + dstart;
            Z(i, d) = U(i, dx) - mu()(dx) - Uhat(i, dx);
         }
      }

      for (int f = 0; f < nfeat; f++)
      {
         Vector zx(dcount), delta_beta(dcount);
         // zx = Z[dstart : dstart + dcount, :] * F[:, f]
         Features->At_mul_Bt(zx, f, Z);
         // TODO: check if sampling randvals for whole [nfeat x dcount] matrix works faster
         auto randvals = Vector::NullaryExpr(dcount, RandNormalGenerator());

         for (int d = 0; d < dcount; d++)
         {
            int dx = d + dstart;
            double A_df = beta_precision(dx) + Lambda(dx, dx) * F_colsq(f);
            double B_df = Lambda(dx, dx) * (zx(d) + beta()(f, dx) * F_colsq(f));
            double A_inv = 1.0 / A_df;
            double beta_new = B_df * A_inv + std::sqrt(A_inv) * randvals(d);
            delta_beta(d) = beta()(f, dx) - beta_new;

            beta()(f, dx) = beta_new;
         }
         // Z[dstart : dstart + dcount, :] += F[:, f] * delta_beta'
         Features->add_Acol_mul_bt(Z, f, delta_beta);
      }
   }
}

void MacauOnePrior::sample_mu_lambda(const Matrix &U)
{
   Matrix WI(num_latent(), num_latent());
   WI.setIdentity();
   int nitem = num_item();

   Matrix Udelta(nitem, num_latent());
   #pragma omp parallel for schedule(static)
   for (int i = 0; i < nitem; i++)
   {
      for (int d = 0; d < num_latent(); d++)
      {
         Udelta(i,d) = U(i,d) - Uhat(i,d);
      }
   }
   std::tie(mu(), Lambda) = CondNormalWishart(Udelta, Vector::Constant(num_latent(), 0.0), 2.0, WI, num_latent());
}

void MacauOnePrior::sample_beta_precision()
{
   double beta_precision_a = beta_precision_a0 + num_feat() / 2.0;
   Vector beta_precision_b = Vector::Constant(num_latent(), beta_precision_b0);
   #pragma omp parallel
   {
      Vector tmp(num_latent());
      tmp.setZero();
      #pragma omp for schedule(static)
      for (int f = 0; f < num_feat(); f++)
      {
         for (int d = 0; d < num_latent(); d++)
         {
            tmp(d) += std::pow(beta()(f, d), 2);
         }
      }
      #pragma omp critical
      {
         beta_precision_b += tmp / 2;
      }
   }
   for (int d = 0; d < num_latent(); d++)
   {
      beta_precision(d) = rand_gamma(beta_precision_a, 1.0 / beta_precision_b(d));
   }
}

std::ostream& MacauOnePrior::status(std::ostream &os, std::string indent) const
{
   os << indent << "  " << m_name << ": Beta = " << beta().norm() << std::endl;
   return os;
}
} // end namespace smurff
