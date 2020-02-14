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
#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/Utils/Distribution.h>

#include <SmurffCpp/Model.h>

#include <SmurffCpp/VMatrixIterator.hpp>
#include <SmurffCpp/ConstVMatrixIterator.hpp>
#include <SmurffCpp/VMatrixExprIterator.hpp>
#include <SmurffCpp/ConstVMatrixExprIterator.hpp>

#include <Utils/Error.h>
#include <SmurffCpp/Utils/Step.h>

#include <SmurffCpp/IO/GenericIO.h>

namespace smurff {

Model::Model()
   : m_num_latent(-1), m_dims(0)
{
}

//num_latent - size of latent dimention
//dims - dimentions of train data
//init_model_type - samples initialization type
void Model::init(int num_latent, const PVec<>& dims, ModelInitTypes model_init_type, bool save_model, bool aggregate)
{
   m_num_latent = num_latent;
   m_dims = dims;
   m_collect_aggr = aggregate;
   m_save_model = save_model;
   m_num_aggr = std::vector<int>(dims.size(), 0);

   for(size_t i = 0; i < dims.size(); ++i)
   {
      std::shared_ptr<Matrix> mat(new Matrix(m_num_latent, dims[i]));

      switch(model_init_type)
      {
      case ModelInitTypes::random:
         bmrandn(*mat);
         break;
      case ModelInitTypes::zero:
         mat->setZero();
         break;
      default:
         {
            THROWERROR("Invalid model init type");
         }
      }

      m_factors.push_back(mat);

      if (aggregate)
      {
         std::shared_ptr<Matrix> aggr_sum(new Matrix(m_num_latent, dims[i]));
         aggr_sum->setZero();
         m_aggr_sum.push_back(aggr_sum);
         std::shared_ptr<Matrix> aggr_dot(new Matrix(m_num_latent * m_num_latent, dims[i]));
         aggr_dot->setZero();
         m_aggr_dot.push_back(aggr_dot);
      }
   }

   m_link_matrices.resize(nmodes());
   m_mus.resize(nmodes());

   Pcache.init(Array1D::Ones(m_num_latent));
}

void Model::setLinkMatrix(int mode, 
   std::shared_ptr<Matrix> link_matrix,
   std::shared_ptr<Vector> mu
   )
{
   m_link_matrices.at(mode) = link_matrix;
   m_mus.at(mode) = mu;
}

double Model::predict(const PVec<> &pos) const
{
   if (nmodes() == 2)
   {
      return col(0, pos[0]).dot(col(1, pos[1]));
   }

   auto &P = Pcache.local();
   P.setOnes();
   for(uint32_t d = 0; d < nmodes(); ++d)
      P *= col(d, pos.at(d)).array();
   return P.sum();
}

const Matrix &Model::U(uint32_t f) const
{
   return *m_factors.at(f);
}

Matrix &Model::U(uint32_t f)
{
   return *m_factors[f];
}

VMatrixIterator<Matrix> Model::Vbegin(std::uint32_t mode)
{
   return VMatrixIterator<Matrix>(shared_from_this(), mode, 0);
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

Matrix::ConstColXpr Model::col(int f, int i) const
{
   return U(f).col(i);
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
      [](const int &a, const std::shared_ptr<Matrix> &b) { return a + b->cols(); });
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

   const auto &r = col(m, i);
   Matrix cov = (r * r.transpose());
   m_aggr_sum.at(m)->col(i) += r;
   m_aggr_dot.at(m)->col(i) += Eigen::Map<Vector>(cov.data(), nlatent() * nlatent());
}

void Model::updateAggr(int m)
{
   m_num_aggr.at(m)++;
}

void Model::save(std::shared_ptr<const Step> sf) const
{
   sf->putModel(m_factors);
   for (std::uint64_t m = 0; m < nmodes(); ++m)
   {
      if (m_collect_aggr && m_save_aggr)
      {
         double n = m_num_aggr.at(m);

         Matrix &Usum = *m_aggr_sum.at(m);
         Matrix &Uprod = *m_aggr_dot.at(m);

         Matrix mu = Matrix::Zero(Usum.rows(), Usum.cols());
         // inverse of the covariance
         Matrix prec = Matrix::Zero(Uprod.rows(), Uprod.cols());

         // calculate real mu and Lambda
         for (int i = 0; i < U(m).cols(); i++)
         {
            Vector sum = Usum.col(i);
            Matrix prod = Eigen::Map<Matrix>(Uprod.col(i).data(), nlatent(), nlatent());
            Matrix prec_i = ((prod - (sum * sum.transpose() / n)) / (n - 1)).inverse();

            prec.col(i) = Eigen::Map<Vector>(prec_i.data(), nlatent() * nlatent());
            mu.col(i) = sum / n;
         }

         sf->putPostMuLambda(m, mu, prec);
      }
   }
}

void Model::restore(std::shared_ptr<const Step> sf, int skip_mode)
{
   unsigned nmodes = sf->getNModes();
   m_factors.clear();
   m_dims = PVec<>(nmodes);

   for (std::uint64_t i = 0; i < nmodes; ++i)
   {
      std::shared_ptr<Matrix> U;
      if ((int)i != skip_mode)
      {
         U = sf->getModel(i);
         m_dims.at(i) = U->cols();
         m_num_latent = U->rows();
      }
      else
      {
         m_dims.at(i) = -1;
      }
      m_factors.push_back(U);
   }

   m_link_matrices.resize(nmodes);
   m_mus.resize(nmodes);

   for(int i=0; i<nmodes; ++i)
   {
       std::shared_ptr<Vector> mu = sf->getMu(i);
       std::shared_ptr<Matrix> beta = sf->getLinkMatrix(i);
       setLinkMatrix(i, beta, mu);
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
         << U(d).rowwise().mean().transpose()
         << std::endl;

   return os;
}

Matrix::ConstBlockXpr SubModel::U(int f) const
{
   const Matrix &u = m_model.U(f); //force const
   return u.block(0, m_off.at(f), m_model.nlatent(), m_dims.at(f));
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
