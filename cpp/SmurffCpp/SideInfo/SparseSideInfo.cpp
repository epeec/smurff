#include "SparseSideInfo.h"
#include "linop.h"

#include <SmurffCpp/Utils/MatrixUtils.h>

#include <vector>

namespace smurff {

SparseSideInfo::SparseSideInfo(const DataConfig &mc) {
    F = mc.getSparseMatrixData();
    Ft = F.transpose();
    m_sideinfo = matrix_utils::to_af(F);
    m_sideinfo_t = m_sideinfo.T();
}

SparseSideInfo::~SparseSideInfo() {}


int SparseSideInfo::cols() const
{
   return F.cols();
}

int SparseSideInfo::rows() const
{
   return F.rows();
}

std::ostream& SparseSideInfo::print(std::ostream &os) const
{
   double percent = 100.8 * (double)F.nonZeros() / (double)F.rows() / (double) F.cols();
   os << "SparseDouble " << F.nonZeros() << " [" << F.rows() << ", " << F.cols() << "] ("
      << percent << "%)" << std::endl;
   return os;
}

bool SparseSideInfo::is_dense() const
{
   return false;
}

af::array SparseSideInfo::arr() const
{
    return m_sideinfo;
}

void SparseSideInfo::compute_uhat(Matrix& uhat, Matrix& beta)
{
    COUNTER("compute_uhat");
    uhat = F * beta;
}

void SparseSideInfo::At_mul_A(Matrix& out)
{
    COUNTER("At_mul_A");
    out = Ft * F;
}

void SparseSideInfo::A_mul_B(const Matrix& A, Matrix &out)
{
    COUNTER("A_mul_B");
    out = Ft * A;
}

int SparseSideInfo::solve_blockcg(Matrix& X, double reg, Matrix& B, double tol, const int blocksize, const int excess, bool throw_on_cholesky_error)
{
    COUNTER("solve_blockcg");
    return linop::solve_blockcg(X, *this, reg, B, tol, blocksize, excess, throw_on_cholesky_error);
}

Vector SparseSideInfo::col_square_sum()
{
    COUNTER("col_square_sum");
    // component-wise square
    auto E = F.unaryExpr([](const float_type &d) { return d * d; });
    // col-wise sum
    return E.transpose() * Vector::Ones(E.rows()).transpose();
}

// Y = X[:,row]' * B'
void SparseSideInfo::At_mul_Bt(Vector& Y, const int row, Matrix& B)
{
    COUNTER("At_mul_Bt");
    Y = Ft.row(row) * B;
}

// computes Z += A[:,row] * b', where a and b are vectors
void SparseSideInfo::add_Acol_mul_bt(Matrix& Z, const int col, Vector& b)
{
    COUNTER("add_Acol_mul_bt");
    Z += F.col(col) * b;
}
} // end namespace smurff
