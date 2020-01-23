#include "SparseSideInfo.h"
#include "linop.h"

#include <SmurffCpp/Utils/MatrixUtils.h>

#include <vector>

namespace smurff {

SparseSideInfo::SparseSideInfo(const std::shared_ptr<MatrixConfig> &mc) {
    F = matrix_utils::sparse_to_eigen(*mc);
    Ft = F.transpose();
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

void SparseSideInfo::compute_uhat(Matrix& uhat, Matrix& beta)
{
    COUNTER("compute_uhat");
    uhat = beta * (Ft);
}

void SparseSideInfo::At_mul_A(Matrix& out)
{
    COUNTER("At_mul_A");
    out = Ft * F;
}

Matrix SparseSideInfo::A_mul_B(Matrix& A)
{
    COUNTER("A_mul_B");
    return (A * F);
}

int SparseSideInfo::solve_blockcg(Matrix& X, double reg, Matrix& B, double tol, const int blocksize, const int excess, bool throw_on_cholesky_error)
{
    COUNTER("solve_blockcg");
    return linop::solve_blockcg(X, *this, reg, B, tol, blocksize, excess, throw_on_cholesky_error);
#if 0
    int iter1, iter2;
    Matrix X1 = X;
    {
        COUNTER("eigen_cg");
        linop::AtA A(F, reg);
        Eigen::ConjugateGradient<linop::AtA, Eigen::Lower | Eigen::Upper> cg;
        cg.setTolerance(tol);
        cg.compute(A);
        X1 = cg.solve(B.transpose()).transpose();
        iter1 = cg.iterations();
        SHOW(iter1);
        SHOW((X1 - B).norm());
    }

    Matrix X2 = X;
    {
        COUNTER("smurff_cg");
        iter2 = linop::solve_blockcg(X2, *this, reg, B, tol, blocksize, excess, throw_on_cholesky_error);
        SHOW(iter2);
        SHOW((X2 - B).norm());
    }

    SHOW((X2 - X1).norm());

    return iter1;
#endif
}

Vector SparseSideInfo::col_square_sum()
{
    COUNTER("col_square_sum");
    // component-wise square
    auto E = F.unaryExpr([](const double &d) { return d * d; });
    // col-wise sum
    return E.transpose() * Vector::Ones(E.rows());
}

// Y = X[:,col]' * B'
void SparseSideInfo::At_mul_Bt(Vector& Y, const int col, Matrix& B)
{
    COUNTER("At_mul_Bt");
    auto out = Ft.block(col, 0, col + 1, Ft.cols()) * B.transpose();
    Y = out.transpose();
}

// computes Z += A[:,col] * b', where a and b are vectors
void SparseSideInfo::add_Acol_mul_bt(Matrix& Z, const int col, Vector& b)
{
    COUNTER("add_Acol_mul_bt");
    Z += (F.col(col) * b.transpose()).transpose();
}
} // end namespace smurff
