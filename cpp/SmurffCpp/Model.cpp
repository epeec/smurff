#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <memory>
#include <cmath>
#include <signal.h>

#include <SmurffCpp/Types.h>

#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/Utils/Distribution.h>

#include <SmurffCpp/Model.h>

#include <SmurffCpp/VMatrixIterator.hpp>
#include <SmurffCpp/ConstVMatrixIterator.hpp>
#include <SmurffCpp/VMatrixExprIterator.hpp>
#include <SmurffCpp/ConstVMatrixExprIterator.hpp>

#include <Utils/Error.h>
#include <SmurffCpp/Utils/SaveState.h>

namespace smurff {

Model::Model()
   : m_num_latent(-1), m_dims(0)
{
}

//num_latent - size of latent dimension
//dims - dimentions of train data
//init_model_type - samples initialization type
void Model::init(int num_latent, const PVec<>& dims, ModelInitTypes model_init_type, bool save_model, bool aggregate)
{
   size_t nmodes = dims.size();

   m_num_latent = num_latent;
   m_dims = dims;
   m_collect_aggr = aggregate;
   m_save_model = save_model;
   m_num_aggr = std::vector<int>(dims.size(), 0);

   m_factors.resize(nmodes);
   m_link_matrices.resize(nmodes);
   m_mus.resize(nmodes);

   for(size_t i = 0; i < nmodes; ++i)
   {
      Matrix& mat = m_factors.at(i);
      mat.resize(dims[i], m_num_latent);

      switch(model_init_type)
      {
      case ModelInitTypes::random:
         rand_normal(mat);
         break;
      case ModelInitTypes::zero:
         mat.setZero();
         break;
      default:
         {
            THROWERROR("Invalid model init type");
         }
      }

      if (aggregate)
      {
         m_aggr_sum.push_back(Matrix::Zero(dims[i], m_num_latent));
         m_aggr_dot.push_back(Matrix::Zero(dims[i], m_num_latent * m_num_latent));
      }
   }

   Pcache.init(Array1D::Ones(m_num_latent));
}

Matrix &Model::getLinkMatrix(int mode)
{
   return m_link_matrices.at(mode);
}

Vector &Model::getMu(int mode)
{
   return m_mus.at(mode);
}

const Matrix &Model::getLinkMatrix(int mode) const
{
   return m_link_matrices.at(mode);
}

const Vector &Model::getMu(int mode) const
{
   return m_mus.at(mode);
}

double Model::predict(const PVec<> &pos) const
{
   if (nmodes() == 2)
   {
      return row(0, pos[0]).dot(row(1, pos[1]));
   }

   auto &P = Pcache.local();
   P.setOnes();
   for(uint32_t d = 0; d < nmodes(); ++d)
      P *= row(d, pos.at(d)).array();
   return P.sum();
}

const Matrix &Model::U(uint32_t f) const
{
   return m_factors.at(f);
}

Matrix &Model::U(uint32_t f)
{
   return m_factors[f];
}

VMatrixIterator<Matrix> Model::Vbegin(std::uint32_t mode)
{
   return VMatrixIterator<Matrix>(this, mode, 0);
}

VMatrixIterator<Matrix> Model::Vend()
{
   return VMatrixIterator<Matrix>(m_factors.size());
}

ConstVMatrixIterator<Matrix> Model::CVbegin(std::uint32_t mode) const
{
   return ConstVMatrixIterator<Matrix>(this, mode, 0);
}

ConstVMatrixIterator<Matrix> Model::CVend() const
{
   return ConstVMatrixIterator<Matrix>(m_factors.size());
}

Matrix::ConstRowXpr Model::row(int f, int i) const
{
   return U(f).row(i);
}

std::uint64_t Model::nmodes() const
{
   return m_factors.size();
}

int Model::nlatent() const
{
   return m_num_latent;
}

int Model::nsamples() const
{
   return std::accumulate(m_factors.begin(), m_factors.end(), 0,
      [](const int &a, const Matrix &b) { return a + b.rows(); });
}

const PVec<>& Model::getDims() const
{
   return m_dims;
}

SubModel Model::full()
{
   return SubModel(*this);
}

void Model::updateAggr(int m, int i)
{
   if (!m_collect_aggr) return;

   const auto &r = row(m, i);
   Matrix cov = (r.transpose() * r);
   m_aggr_sum.at(m).row(i) += r;
   m_aggr_dot.at(m).row(i) += Eigen::Map<Vector>(cov.data(), nlatent() * nlatent());
}

void Model::updateAggr(int m)
{
   m_num_aggr.at(m)++;
}

void Model::save(SaveState &sf) const
{
   sf.putModel(m_factors);
   for (std::uint64_t m = 0; m < nmodes(); ++m)
   {
      sf.putLinkMatrix(m, m_link_matrices.at(m));
      sf.putMu(m, m_mus.at(m));

      if (m_collect_aggr && m_save_aggr)
      {
         double n = m_num_aggr.at(m);

         const Matrix &Usum = m_aggr_sum.at(m);
         const Matrix &Uprod = m_aggr_dot.at(m);

         Matrix mu = Matrix::Zero(Usum.rows(), Usum.cols());
         // inverse of the covariance
         Matrix prec = Matrix::Zero(Uprod.rows(), Uprod.cols());

         // calculate real mu and Lambda
         for (int i = 0; i < U(m).rows(); i++)
         {
            Vector sum = Usum.row(i);
            Matrix prod = Eigen::Map<const Matrix>(Uprod.row(i).data(), nlatent(), nlatent());
            Matrix prec_i = ((prod - (sum.transpose() * sum / n)) / (n - 1)).inverse();

            prec.row(i) = Eigen::Map<Vector>(prec_i.data(), nlatent() * nlatent());
            mu.row(i) = sum / n;
         }

         sf.putPostMuLambda(m, mu, prec);
      }
   }
}

void Model::restore(const SaveState &sf, int skip_mode)
{
   unsigned nmodes = sf.getNModes();
   m_factors.clear();
   m_dims = PVec<>(nmodes);
   m_factors.resize(nmodes);

   for (std::uint64_t i = 0; i < nmodes; ++i)
   {
      if ((int)i != skip_mode)
      {
         auto &U = m_factors.at(i);
         sf.readModel(i, U);
         m_dims.at(i) = U.rows();
         m_num_latent = U.cols();
      }
      else
      {
         m_dims.at(i) = -1;
      }
   }

   m_link_matrices.resize(nmodes);
   m_mus.resize(nmodes);

   for(int i=0; i<nmodes; ++i)
   {
      sf.readLinkMatrix(i, m_link_matrices.at(i));
      sf.readMu(i, m_mus.at(i));
   }

   Pcache.init(Array1D::Ones(m_num_latent));
}

std::ostream& Model::info(std::ostream &os, std::string indent) const
{
   os << indent << "Num-latents: " << m_num_latent << std::endl;
   return os;
}

std::ostream& Model::status(std::ostream &os, std::string indent) const
{
   os << indent << "  Umean: " << std::endl;
   for(std::uint64_t d = 0; d < nmodes(); ++d)
      os << indent << "    U(" << d << ").colwise().mean: "
         << U(d).colwise().mean()
         << std::endl;

   return os;
}

Matrix::ConstBlockXpr SubModel::U(int f) const
{
   const Matrix &u = m_model.U(f); //force const
   return u.block(m_off.at(f), 0, m_dims.at(f), m_model.nlatent());
}

ConstVMatrixExprIterator<Matrix::ConstBlockXpr> SubModel::CVbegin(std::uint32_t mode) const
{
   return ConstVMatrixExprIterator<Matrix::ConstBlockXpr>(&m_model, m_off, m_dims, mode, 0);
}

ConstVMatrixExprIterator<Matrix::ConstBlockXpr> SubModel::CVend() const
{
   return ConstVMatrixExprIterator<Matrix::ConstBlockXpr>(m_model.nmodes());
}
} // end namespace smurff
