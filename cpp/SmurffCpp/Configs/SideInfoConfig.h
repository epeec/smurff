#pragma once

#include <vector>
#include <memory>
#include <string>

#include "DataConfig.h"

namespace smurff
{
   class HDF5Group;

   class SideInfoConfig : public DataConfig
   {
   public:
      static const bool DIRECT_DEFAULT_VALUE;
      static const double BETA_PRECISION_DEFAULT_VALUE;
      static const double TOL_DEFAULT_VALUE;

   private:
      double m_tol = TOL_DEFAULT_VALUE;
      bool m_direct = DIRECT_DEFAULT_VALUE;
      bool m_throw_on_cholesky_error = false;

   public:
      SideInfoConfig() {}; //empty
      SideInfoConfig(const Matrix &, const NoiseConfig &);
      SideInfoConfig(const SparseMatrix &, const NoiseConfig &);

   public:
      double getTol() const
      {
         return m_tol;
      }

      void setTol(double value)
      {
         m_tol = value;
      }

      bool getDirect() const
      {
         return m_direct;
      }

      void setDirect(bool value)
      {
         m_direct = value;
      }

      bool getThrowOnCholeskyError() const
      {
         return m_throw_on_cholesky_error;
      }

      void setThrowOnCholeskyError(bool value)
      {
         m_throw_on_cholesky_error = value;
      }

   public:
      void save(HDF5Group& writer, std::size_t prior_index) const;
      bool restore(const HDF5Group& reader, std::size_t prior_index);

   };
}
