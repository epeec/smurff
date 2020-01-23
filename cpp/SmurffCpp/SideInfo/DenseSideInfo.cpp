#include "DenseSideInfo.h"
#include "linop.h"

namespace smurff {

DenseSideInfo::DenseSideInfo(const std::shared_ptr<MatrixConfig> &side_info)
{
   m_side_info = std::make_shared<Matrix>(matrix_utils::dense_to_eigen(*side_info));
}

int DenseSideInfo::cols() const
{
   return m_side_info->cols();
}

int DenseSideInfo::rows() const
{
   return m_side_info->rows();
}

std::ostream& DenseSideInfo::print(std::ostream &os) const
{
   os << "DenseDouble [" << m_side_info->rows() << ", " << m_side_info->cols() << "]" << std::endl;
   return os;
}

bool DenseSideInfo::is_dense() const
{
   return true;
}

void DenseSideInfo::compute_uhat(Matrix& uhat, Matrix& beta)
{
   uhat = beta * m_side_info->transpose();
}

void DenseSideInfo::At_mul_A(Matrix& out)
{
   out = m_side_info->transpose() * *m_side_info;
}

Matrix DenseSideInfo::A_mul_B(Matrix& A)
{
   return A * *m_side_info;
}

int DenseSideInfo::solve_blockcg(Matrix& X, double reg, Matrix& B, double tol, const int blocksize, const int excess, bool throw_on_cholesky_error)
{
   return linop::solve_blockcg(X, *m_side_info, reg, B, tol, blocksize, excess, throw_on_cholesky_error);
}

Vector DenseSideInfo::col_square_sum()
{
    return m_side_info->array().square().colwise().sum();
}


void DenseSideInfo::At_mul_Bt(Vector& Y, const int col, Matrix& B)
{
   Y = B * m_side_info->col(col);
}

void DenseSideInfo::add_Acol_mul_bt(Matrix& Z, const int col, Vector& b)
{
   Z += (m_side_info->col(col) * b.transpose()).transpose();
}

std::shared_ptr<Matrix> DenseSideInfo::get_features()
{
   return m_side_info;
}
} // end namespace smurff
