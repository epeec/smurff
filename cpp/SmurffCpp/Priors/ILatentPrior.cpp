#include "ILatentPrior.h"
#include <Utils/counters.h>

namespace smurff {

ILatentPrior::ILatentPrior(std::shared_ptr<Session> session, uint32_t mode, std::string name)
   : m_session(session), m_mode(mode), m_name(name)
{

}

void ILatentPrior::init()
{
   rrs.init(Vector::Zero(num_latent()));
   MMs.init(Matrix::Zero(num_latent(), num_latent()));

   //this is some new initialization
   init_Usum();
}

const Model& ILatentPrior::model() const
{
   return getSession().model();
}

Model& ILatentPrior::model()
{
   return getSession().model();
}

double ILatentPrior::predict(const PVec<> &pos) const
{
    return model().predict(pos);
}

Data& ILatentPrior::data() const
{
   return getSession().data();
}

INoiseModel& ILatentPrior::noise()
{
   return data().noise();
}

Matrix &ILatentPrior::U()
{
   return model().U(m_mode);
}

const Matrix &ILatentPrior::U() const
{
   return model().U(m_mode);
}

//return V matrices in the model opposite to mode
VMatrixIterator<Matrix> ILatentPrior::Vbegin()
{
   return model().Vbegin(m_mode);
}

VMatrixIterator<Matrix> ILatentPrior::Vend()
{
   return model().Vend();
}

ConstVMatrixIterator<Matrix> ILatentPrior::CVbegin() const
{
   return model().CVbegin(m_mode);
}

ConstVMatrixIterator<Matrix> ILatentPrior::CVend() const
{
   return model().CVend();
}

int ILatentPrior::num_latent() const
{
   return model().nlatent();
}

int ILatentPrior::num_item() const
{
   return model().U(m_mode).cols();
}

std::ostream &ILatentPrior::info(std::ostream &os, std::string indent)
{
   os << indent << m_mode << ": " << m_name << std::endl;
   return os;
}

bool ILatentPrior::run_slave()
{
   return false;
}

void ILatentPrior::sample_latents()
{
   COUNTER("sample_latents");
   data().update_pnm(model(), m_mode);

   // for effiency, we keep + update Ucol and UUcol by every thread
   thread_vector<Vector> Ucol(Vector::Zero(num_latent()));
   thread_vector<Matrix> UUcol(Matrix::Zero(num_latent(), num_latent()));

   #pragma omp parallel for schedule(guided)
   for(int n = 0; n < U().cols(); n++)
   #pragma omp task
   {
      COUNTER("sample_latent");
      sample_latent(n);
      const auto &col = U().col(n);
      Ucol.local().noalias() += col;
      UUcol.local().noalias() += col * col.transpose();

      if (getSession().inSamplingPhase())
         model().updateAggr(m_mode, n);
   }

   if (getSession().inSamplingPhase())
      model().updateAggr(m_mode);

   Usum  = Ucol.combine();
   UUsum = UUcol.combine();
}

bool ILatentPrior::save(std::shared_ptr<const Step> sf) const
{
    return false;
}

void ILatentPrior::restore(std::shared_ptr<const Step> sf)
{
    init_Usum();
    update_prior();
}

void ILatentPrior::init_Usum()
{
    Usum = U().rowwise().sum();
    UUsum = U() * U().transpose(); 
}
} // end namespace smurff
