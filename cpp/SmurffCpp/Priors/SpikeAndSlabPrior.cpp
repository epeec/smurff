#include "SpikeAndSlabPrior.h"

#include <Utils/Error.h>

namespace smurff {

SpikeAndSlabPrior::SpikeAndSlabPrior(TrainSession &trainSession, uint32_t mode)
   : NormalOnePrior(trainSession, mode, "SpikeAndSlabPrior")
{

}

void SpikeAndSlabPrior::init()
{
   NormalOnePrior::init();

   const int K = num_latent();
   const int D = num_item();
   const int nview = data().nview(m_mode);
   
   THROWERROR_ASSERT(D > 0);

   Zcol.init(Matrix::Zero(nview,K));
   W2col.init(Matrix::Zero(nview,K));

   //-- prior params
   Zkeep = Array2D::Constant(nview, K, D);

   alpha = Array2D::Ones(nview,K);
   log_alpha.resize(nview, K);
   log_alpha = alpha.log();

   r = Array2D::Constant(nview,K,.5);
   log_r.resize(nview, K);
   log_r = - r.log() + (Array2D::Ones(nview, K) - r).log();
}

void SpikeAndSlabPrior::update_prior()
{
   const int nview = data().nview(m_mode);
   const int K = num_latent();
   
   Zkeep = Zcol.combine();
   auto W2c = W2col.combine();

   // update hyper params (alpha and r) (per view)
   for(int v=0; v<nview; ++v) {
       const int D = data().view_size(m_mode, v);
       r.row(v) = ( Zkeep.row(v).array() + prior_beta ) / ( D + prior_beta * D ) ;
       auto ww = W2c.row(v).array() / 2 + prior_beta_0;
       auto tmpz = Zkeep.row(v).array() / 2 + prior_alpha_0 ;
       alpha.row(v) = tmpz.binaryExpr(ww, [](float_type a, float_type b)->float_type {
               return rgamma(a, 1/b) + 1e-7;
       });
   }

   Zcol.reset();
   W2col.reset(); 

   log_alpha = alpha.log();
   log_r = - r.log() + (Array2D::Ones(nview, K) - r).log();
}

void SpikeAndSlabPrior::restore(const SaveState &sf)
{
  const int K = num_latent();
  const int nview = data().nview(m_mode);

  ILatentPrior::restore(sf);

  //compute Zcol
  int d = 0;
  Array2D Z(Array2D::Zero(nview,K));
  Array2D W2(Array2D::Zero(nview,K));
  for(int v=0; v<data().nview(m_mode); ++v) 
  {
      for(int i=0; i<data().view_size(m_mode, v); ++i, ++d)
      {
        for(int k=0; k<K; ++k) if (U()(k,d) != 0) Z(v,k)++;
        W2.row(v) += U().row(d).array().square(); 
      }
  }
  THROWERROR_ASSERT(d == num_item());

  Zcol.reset();
  W2col.reset();
  Zcol.local() = Z;
  W2col.local() = W2;

  update_prior();
}

std::pair<float_type, float_type> SpikeAndSlabPrior::sample_latent(int d, int k, const Matrix& XX, const Vector& yX)
{
    const int v = data().view(m_mode, d);
    float_type mu, lambda;

    Matrix aXX = alpha.matrix().row(v).asDiagonal();
    aXX += XX;
    std::tie(mu, lambda) = NormalOnePrior::sample_latent(d, k, aXX, yX);

    auto Urow = U().row(d);
    float_type z1 = log_r(v,k) -  0.5 * (lambda * mu * mu - std::log(lambda) + log_alpha(v,k));
    float_type z = 1 / (1 + exp(z1));
    float_type p = rand_unif(0,1);
    if (Zkeep(v,k) > 0 && p < z) {
        Zcol.local()(v,k)++;
        W2col.local()(v,k) += Urow(k) * Urow(k);
    } else {
        Urow(k) = .0;
    }

    return std::make_pair(mu, lambda);
}

std::ostream &SpikeAndSlabPrior::status(std::ostream &os, std::string indent) const
{
   const int V = data().nview(m_mode);
   for(int v=0; v<V; ++v) 
   {
       int Zcount = (Zkeep.row(v).array() > 0).count();
       os << indent << m_name << ": Z[" << v << "] = " << Zcount << "/" << num_latent() << std::endl;
   }
   return os;
}
} // end namespace smurff
