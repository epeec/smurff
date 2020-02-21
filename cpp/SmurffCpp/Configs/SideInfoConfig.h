#pragma once

#include <vector>
#include <memory>
#include <string>

#include "MatrixConfig.h"

namespace smurff
{
   class ConfigFile;

   class SideInfoConfig : public DataConfig
   {
   public:
      static double BETA_PRECISION_DEFAULT_VALUE;
      static double TOL_DEFAULT_VALUE;
   private:
      double m_tol;
      bool m_direct;
      bool m_throw_on_cholesky_error;

   public:
      SideInfoConfig();

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
      void save(ConfigFile& writer, std::size_t prior_index) const;
      bool restore(const ConfigFile& reader, std::size_t prior_index);

   public:
      std::shared_ptr<Data> create(std::shared_ptr<IDataCreator> creator) const override;

   public:
      void write(std::shared_ptr<IDataWriter> writer) const override;
   };
}
