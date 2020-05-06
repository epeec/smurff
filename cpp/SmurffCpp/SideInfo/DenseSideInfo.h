#pragma once

#include <memory>

#include <SmurffCpp/Types.h>

#include <SmurffCpp/Configs/DataConfig.h>

#include "ISideInfo.h"


namespace smurff {

   class DenseSideInfo : public ISideInfo
   {
   private:
      Matrix m_side_info, m_side_info_t;

   public:
      DenseSideInfo(const DataConfig &);

   public:
      int cols() const override;

      int rows() const override;

   public:
      std::ostream& print(std::ostream &os) const override;

      bool is_dense() const override;

   public:
      //linop

      void compute_uhat(Matrix& uhat, Matrix& beta) override;

      void At_mul_A(Matrix& out) override;

      void A_mul_B(const Matrix& A, Matrix &out) override;

      int solve_blockcg(Matrix& X, double reg, Matrix& B, double tol, const int blocksize, const int excess, bool throw_on_cholesky_error = false) override;

      Vector col_square_sum() override;

      void At_mul_Bt(Vector& Y, const int row, Matrix& B) override;

      void add_Acol_mul_bt(Matrix& Z, const int row, Vector& b) override;

      //only for tests
   public:
      const Matrix &get_features();
   };

}
