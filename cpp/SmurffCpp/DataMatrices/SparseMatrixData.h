#pragma once

#include "FullMatrixData.hpp"

namespace smurff
{
   class SparseMatrixData : public FullMatrixData<SparseMatrix >
   {
   public:
      SparseMatrixData(SparseMatrix Y);

      void getMuLambda(const SubModel& model, std::uint32_t mode, int d, Vector& rr, Matrix& MM) const override;

   public:
      double train_rmse(const SubModel& model) const override;

   public:
      double var_total() const override;
      
      double sumsq(const SubModel& model) const override;
  };
}
