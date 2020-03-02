#include "NormalOnePrior.h"

namespace smurff {

NormalOnePrior::NormalOnePrior(TrainSession &trainSession, uint32_t mode, std::string name)
   : ILatentPrior(trainSession, mode, name)
{
}

void NormalOnePrior::init()
{
   //does not look that there was such init previously
   ILatentPrior::init();

   const int K = num_latent();

   mu().resize(K);
   mu().setZero(); 

   Lambda.resize(K, K);
   Lambda.setIdentity();
   Lambda *= 10.;

   // parameters of Inv-Whishart distribution
   WI.resize(K, K);
   WI.setIdentity();
   mu0.resize(K);
   mu0.setZero();
   b0 = 2;
   df = K;
}

const Vector NormalOnePrior::fullMu(int n) const
{
   return mu();
}

void NormalOnePrior::update_prior()
{
    std::tie(mu(), Lambda) = CondNormalWishart(num_item(), getUUsum(), getUsum(), mu0, b0, WI, df);
}

void NormalOnePrior::sample_latent(int d)
{
   const int K = num_latent();

   Matrix XX = Matrix::Zero(K, K);
   Vector yX = Vector::Zero(K);

   data().getMuLambda(model(), m_mode, d, yX, XX);

   // add hyperparams
   yX.noalias() += Lambda * mu();
   XX.noalias() += Lambda;

   for(int k=0;k<K;++k) sample_latent(d, k, XX, yX);
}
 
std::pair<float_type,float_type> NormalOnePrior::sample_latent(int d, int k, const Matrix& XX, const Vector& yX)
{
    auto Urow = U().row(d);
    float_type lambda = XX(k,k);
    float_type mu = (1/lambda) * (yX(k) - Urow * XX.row(k).transpose() + Urow(k) * XX(k,k));
    Urow(k) = mu + randn() / sqrt(lambda);
    return std::make_pair(mu, lambda);
}

std::ostream &NormalOnePrior::status(std::ostream &os, std::string indent) const
{
   return os;
}
} // end namespace smurff
