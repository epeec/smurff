#pragma once

#include "MatrixDataTempl.hpp"

namespace smurff
{
   class ScarceMatrixData : public MatrixDataTempl<SparseMatrix >
   {
   private:
      int num_empty[2] = {0,0};

   public:
      ScarceMatrixData(SparseMatrix Y);

   public:
      void init_pre() override;
      
      double train_rmse(const SubModel& model) const override;

      std::ostream& info(std::ostream& os, std::string indent) override;

      void getMuLambda(const SubModel& model, std::uint32_t mode, int d, Vector& rr, Matrix& MM) const override;
      void update_pnm(const SubModel& model, std::uint32_t mode) override;

      std::uint64_t nna() const override;

   public:
      double var_total() const override;
      
      double sumsq(const SubModel& model) const override;
   };
}
