#pragma once

#include <iostream>

#include <SmurffCpp/Types.h>

namespace smurff {

   class ISideInfo
   {
   public:
      virtual ~ISideInfo() {}

      virtual int cols() const = 0;

      virtual int rows() const = 0;

      virtual std::ostream& print(std::ostream &os) const = 0;

      virtual bool is_dense() const = 0;

   public:
      //linop

      virtual void compute_uhat(Matrix& uhat, Matrix& beta) = 0;

      virtual void At_mul_A(Matrix& out) = 0;

      virtual void A_mul_B(const Matrix& A, Matrix &out) = 0;

      virtual int solve_blockcg(Matrix& X, double reg, Matrix& B, double tol, const int blocksize, const int excess, bool throw_on_cholesky_error = false) = 0;

      virtual Vector col_square_sum() = 0;

      virtual void At_mul_Bt(Vector& Y, const int row, Matrix& B) = 0;

      virtual void add_Acol_mul_bt(Matrix& Z, const int row, Vector& b) = 0;
   };

}
