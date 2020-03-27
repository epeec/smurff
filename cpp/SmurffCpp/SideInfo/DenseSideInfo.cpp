#include <Utils/Error.h>

#include "DenseSideInfo.h"
#include "linop.h"

namespace smurff {

DenseSideInfo::DenseSideInfo(const DataConfig &side_info)
{
   m_side_info = side_info.getDenseMatrixData();
}

int DenseSideInfo::cols() const
{
   return m_side_info.cols();
}

int DenseSideInfo::rows() const
{
   return m_side_info.rows();
}

std::ostream& DenseSideInfo::print(std::ostream &os) const
{
   os << "DenseDouble [" << rows() << ", " << cols() << "]" << std::endl;
   return os;
}

bool DenseSideInfo::is_dense() const
{
   return true;
}

void DenseSideInfo::compute_uhat(Matrix& uhat, Matrix& beta)
{
   uhat = m_side_info * beta;
}

void DenseSideInfo::At_mul_A(Matrix& out)
{
   out = m_side_info.transpose() * m_side_info;
}

Matrix DenseSideInfo::A_mul_B(Matrix& A)
{
   return m_side_info.transpose() * A;
}

int DenseSideInfo::solve_blockcg(Matrix& X, double reg, Matrix& B, double tol, const int blocksize, const int excess, bool throw_on_cholesky_error)
{
   THROWERROR_NOTIMPL();
}

Vector DenseSideInfo::col_square_sum()
{
    return m_side_info.array().square().colwise().sum();
}


void DenseSideInfo::At_mul_Bt(Vector& Y, const int feat, Matrix& B)
{
   Y = m_side_info.col(feat).transpose() * B;
}

void DenseSideInfo::add_Acol_mul_bt(Matrix& Z, const int row, Vector& b)
{
   Z += m_side_info.col(row) * b;
}

const Matrix &DenseSideInfo::get_features()
{
   return m_side_info;
}
} // end namespace smurff
