#include "SpikeAndSlabPrior.h"

#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/IO/GenericIO.h>
#include <Utils/Error.h>

namespace smurff {

SpikeAndSlabPrior::SpikeAndSlabPrior(std::shared_ptr<Session> session, uint32_t mode)
   : NormalOnePrior(session, mode, "SpikeAndSlabPrior")
{

}

void SpikeAndSlabPrior::init()
{
   NormalOnePrior::init();

   const int K = num_latent();
   const int D = num_item();
   const int nview = data().nview(m_mode);
   
   THROWERROR_ASSERT(D > 0);

   Zcol.init(Matrix::Zero(K,nview));
   W2col.init(Matrix::Zero(K,nview));

   //-- prior params
   Zkeep = Array2D::Constant(K, nview, D);

   alpha = Array2D::Ones(K,nview);
   log_alpha.resize(K, nview);
   log_alpha = alpha.log();

   r = Array2D::Constant(K,nview,.5);
   log_r.resize(K, nview);
   log_r = - r.log() + (Array2D::Ones(K, nview) - r).log();
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
       r.col(v) = ( Zkeep.col(v).array() + prior_beta ) / ( D + prior_beta * D ) ;
       auto ww = W2c.col(v).array() / 2 + prior_beta_0;
       auto tmpz = Zkeep.col(v).array() / 2 + prior_alpha_0 ;
       alpha.col(v) = tmpz.binaryExpr(ww, [](float_type a, float_type b)->float_type {
               return rgamma(a, 1/b) + 1e-7;
       });
   }

   Zcol.reset();
   W2col.reset(); 

   log_alpha = alpha.log();
   log_r = - r.log() + (Array2D::Ones(K, nview) - r).log();
}

void SpikeAndSlabPrior::restore(std::shared_ptr<const Step> sf)
{
  const int K = num_latent();
  const int nview = data().nview(m_mode);

  NormalOnePrior::restore(sf);

  //compute Zcol
  int d = 0;
  Array2D Z(Array2D::Zero(K,nview));
  Array2D W2(Array2D::Zero(K,nview));
  for(int v=0; v<data().nview(m_mode); ++v) 
  {
      for(int i=0; i<data().view_size(m_mode, v); ++i, ++d)
      {
        for(int k=0; k<K; ++k) if (U()(k,d) != 0) Z(k,v)++;
        W2.col(v) += U().col(d).array().square(); 
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

    Matrix aXX = alpha.matrix().col(v).asDiagonal();
    aXX += XX;
    std::tie(mu, lambda) = NormalOnePrior::sample_latent(d, k, aXX, yX);

    auto Ucol = U().col(d);
    float_type z1 = log_r(k,v) -  0.5 * (lambda * mu * mu - std::log(lambda) + log_alpha(k,v));
    float_type z = 1 / (1 + exp(z1));
    float_type p = rand_unif(0,1);
    if (Zkeep(k,v) > 0 && p < z) {
        Zcol.local()(k,v)++;
        W2col.local()(k,v) += Ucol(k) * Ucol(k);
    } else {
        Ucol(k) = .0;
    }

    return std::make_pair(mu, lambda);
}

std::ostream &SpikeAndSlabPrior::status(std::ostream &os, std::string indent) const
{
   const int V = data().nview(m_mode);
   for(int v=0; v<V; ++v) 
   {
       int Zcount = (Zkeep.col(v).array() > 0).count();
       os << indent << m_name << ": Z[" << v << "] = " << Zcount << "/" << num_latent() << std::endl;
   }
   return os;
}
} // end namespace smurff
