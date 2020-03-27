#pragma once

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>
#include <Eigen/IterativeLinearSolvers>

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <Utils/Error.h>
#include <Utils/counters.h>

#include <SmurffCpp/SideInfo/SparseSideInfo.h>

namespace smurff {
namespace linop {
  class AtA;
} }

namespace Eigen {
namespace internal {
  // AtA looks-like a SparseMatrix, so let's inherits its traits:
  template<>
  struct traits<smurff::linop::AtA> :  public Eigen::internal::traits<smurff::SparseMatrix>
  {};
}
}

namespace smurff
{
namespace linop
{

// Example of a matrix-free wrapper from a user type to Eigen's compatible type
// For the sake of simplicity, this example simply wrap a Eigen::SparseMatrix.
class AtA : public Eigen::EigenBase<AtA>
{
public:
  // Required typedefs, constants, and method:
  typedef float_type Scalar;
  typedef float_type RealScalar;
  typedef int StorageIndex;
  enum
  {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = true
  };
  Index outerSize() const { return m_A.cols(); }
  Index innerSize() const { return m_A.cols(); }
  Index rows()      const { return m_A.cols(); }
  Index cols()      const { return m_A.cols(); }
  template <typename Rhs>
  Eigen::Product<AtA, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs> &x) const
  {
    return Eigen::Product<AtA, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }
  // Custom API:
  AtA(const SparseMatrix &A, const SparseMatrix &At, double reg) : m_A(A), m_At(At), m_reg(reg) {}

  const SparseMatrix &m_A;
  const SparseMatrix &m_At;
  double m_reg;
};

} // namespace linop
} // namespace smurff

// Implementation of AtA * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {
  template<typename Rhs>
  struct generic_product_impl<smurff::linop::AtA, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
  : generic_product_impl_base<smurff::linop::AtA,Rhs,generic_product_impl<smurff::linop::AtA,Rhs> >
  {
    typedef typename Product<smurff::linop::AtA,Rhs>::Scalar Scalar;
    template<typename Dest>
    static void scaleAndAddTo(Dest& dst, const smurff::linop::AtA& lhs, const Rhs& rhs, const Scalar& alpha)
    {
      // This method should implement "dst += alpha * lhs * rhs" inplace,
      dst += alpha * ((lhs.m_At * (lhs.m_A * rhs)) + lhs.m_reg * rhs);
    }
  };
}
}

namespace smurff
{
namespace linop
{

inline void makeSymmetric(Matrix &A)
{
  A = A.selfadjointView<Eigen::Lower>();
}

inline void AtA_mul_B(Matrix& out, const SparseSideInfo& A, double reg, const Matrix& B) {
  out.noalias() = (A.Ft * (A.F * B)) + reg * B;
}

//
//-- Solves the system (K' * K + reg * I) * X = B for X for m right-hand sides
//   K = d x n matrix
//   I = n x n identity
//   X = n x m matrix
//   B = n x m matrix
//
template<typename T>
inline int solve_blockcg_1block(Matrix & X, const T & K, double reg, Matrix & B, double tol, bool throw_on_cholesky_error = false) {
  // initialize
  const int nfeat = B.rows();
  const int nrhs  = B.cols();
  double tolsq = tol*tol;

  if (nfeat != K.cols()) {THROWERROR("B.rows() must equal K.cols()");}

  Vector norms(nrhs), inorms(nrhs); 
  norms.setZero();
  inorms.setZero();
  #pragma omp parallel for schedule(static)
  for (int rhs = 0; rhs < nrhs; rhs++) 
  {
    double sumsq = 0.0;
    for (int feat = 0; feat < nfeat; feat++) 
    {
      sumsq += B(feat, rhs) * B(feat, rhs);
    }
    norms(rhs)  = std::sqrt(sumsq);
    inorms(rhs) = 1.0 / norms(rhs);
  }
  Matrix R(nfeat, nrhs);
  Matrix P(nfeat, nrhs);
  Matrix Ptmp(nfeat, nrhs);
  X.setZero();
  // normalize R and P:
  #pragma omp parallel for schedule(static) collapse(2)
  for (int feat = 0; feat < nfeat; feat++) 
  {
    for (int rhs = 0; rhs < nrhs; rhs++) 
    {
      R(feat, rhs) = B(feat, rhs) * inorms(rhs);
      P(feat, rhs) = R(feat, rhs);
    }
  }
  Matrix* RtR = new Matrix(nrhs, nrhs);
  Matrix* RtR2 = new Matrix(nrhs, nrhs);

  Matrix   KP(nfeat, nrhs);
  Matrix KPtP(nrhs, nrhs);
  Matrix A;
  Matrix Psi;

  //A_mul_At_combo(*RtR, R);
  *RtR = R.transpose() * R;
  makeSymmetric(*RtR);

  const int nblocks = (int)ceil(nfeat / 64.0);

  // CG iteration:
  int iter = 0;
  for (iter = 0; iter < 1000; iter++) {
    // KP = K * P
    ////double t1 = tick();
    AtA_mul_B(KP, K, reg, P);
    ////double t2 = tick();

    KPtP = KP.transpose() * P;
    auto chol_KPtP = KPtP.llt();
    THROWERROR_ASSERT_MSG(!throw_on_cholesky_error || chol_KPtP.info() != Eigen::NumericalIssue, "Cholesky Decomposition failed! (Numerical Issue)");
    THROWERROR_ASSERT_MSG(!throw_on_cholesky_error || chol_KPtP.info() != Eigen::InvalidInput, "Cholesky Decomposition failed! (Invalid Input)");
    A = chol_KPtP.solve(*RtR);
    ////double t3 = tick();

    
    #pragma omp parallel for schedule(guided)
    for (int block = 0; block < nblocks; block++) 
    {
      int row = block * 64;
      int brows = std::min(64, nfeat - row);
      // X += A' * P
      X.block(row, 0, brows, nrhs).noalias() += P.block(row, 0, brows, nrhs) * A;
      // R -= A' * KP
      R.block(row, 0, brows, nrhs).noalias() -= KP.block(row, 0, brows, nrhs) * A;
    }
    ////double t4 = tick();

    // convergence check:
    //A_mul_At_combo(*RtR2, R);
    *RtR2 = R.transpose() * R;
    makeSymmetric(*RtR2);

    Vector d = RtR2->diagonal();
    // std::cout << "[ iter " << iter << "] " << std::scientific << d.transpose() << " (max: " << d.maxCoeff() << " > " << tolsq << ")" << std::endl;
    //std::cout << iter << ":" << std::scientific << d.transpose() << std::endl;
    if ( (d.array() < tolsq).all()) {
      break;
    } 

    // Psi = (R R') \ R2 R2'
    auto chol_RtR = RtR->llt();
    THROWERROR_ASSERT_MSG(!throw_on_cholesky_error || chol_RtR.info() != Eigen::NumericalIssue, "Cholesky Decomposition failed! (Numerical Issue)");
    THROWERROR_ASSERT_MSG(!throw_on_cholesky_error || chol_RtR.info() != Eigen::InvalidInput, "Cholesky Decomposition failed! (Invalid Input)");
    Psi  = chol_RtR.solve(*RtR2);
    ////double t5 = tick();

    // P = R + Psi' * P (P and R are already transposed)
    #pragma omp parallel for schedule(guided)
    for (int block = 0; block < nblocks; block++) 
    {
      int row = block * 64;
      int brows = std::min(64, nfeat - row);
      Matrix xtmp(brows, nrhs);
      xtmp = P.block(row, 0, brows, nrhs) * Psi;
      P.block(row, 0, brows, nrhs) = R.block(row, 0, brows, nrhs) + xtmp;
    }

    // R R' = R2 R2'
    std::swap(RtR, RtR2);
    ////double t6 = tick();
    ////printf("t2-t1 = %.3f, t3-t2 = %.3f, t4-t3 = %.3f, t5-t4 = %.3f, t6-t5 = %.3f\n", t2-t1, t3-t2, t4-t3, t5-t4, t6-t5);
  }
  
  if (iter == 1000)
  {
    Vector d = RtR2->diagonal().cwiseSqrt();
    std::cerr << "warning: block_cg: could not find a solution in 1000 iterations; residual: ["
              << d.transpose() << " ].all() > " << tol << std::endl;
  }


  // unnormalizing X:
  #pragma omp parallel for schedule(static) collapse(2)
  for (int feat = 0; feat < nfeat; feat++) 
  {
    for (int rhs = 0; rhs < nrhs; rhs++) 
    {
      X(feat, rhs) *= norms(rhs);
    }
  }
  delete RtR;
  delete RtR2;
  return iter;
}


/** good values for solve_blockcg are blocksize=32 an excess=8 */
template<typename T>
inline int solve_blockcg_impl(Matrix & X, const T & K, double reg, Matrix & B, double tol, const int blocksize, const int excess, bool throw_on_cholesky_error = false) {
  if (B.cols() <= excess + blocksize) {
    return solve_blockcg_1block(X, K, reg, B, tol, throw_on_cholesky_error);
  }
  // split B into blocks of size <blocksize> (+ excess if needed)
  Matrix Xblock, Bblock;
  int max_iter = 0;
  for (int i = 0; i < B.cols(); i += blocksize) {
    int ncols = blocksize;
    if (i + ncols + excess >= B.cols()) {
      ncols = B.cols() - i;
    }
    Bblock.resize(B.rows(), ncols);
    Xblock.resize(X.rows(), ncols);

    Bblock = B.block(0, i, B.rows(), ncols);
    int niter = solve_blockcg_1block(Xblock, K, reg, Bblock, tol, throw_on_cholesky_error);
    max_iter = std::max(niter, max_iter);
    X.block(0, i, X.rows(), ncols) = Xblock;
  }

  return max_iter;
}

template<typename T>
inline int solve_blockcg(Matrix & X, const T & K, double reg, Matrix & B, double tol, const int blocksize, const int excess, bool throw_on_cholesky_error)
{
#if 0
    COUNTER("solve_blockcg");
    return linop::solve_blockcg_impl(X, K, reg, B, tol, blocksize, excess, throw_on_cholesky_error);
#else
    int iter1, iter2;
    Matrix X1 = X;
    {
        COUNTER("eigen_cg");
        linop::AtA A(K.F, K.Ft, reg);
        Eigen::ConjugateGradient<linop::AtA, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> cg;
        cg.setTolerance(tol);
        cg.compute(A);
        X1 = cg.solve(B);
        iter1 = cg.iterations();
        SHOW(iter1);
        SHOW((X1 - B).norm());
    }

    Matrix X2 = X;
    {
        COUNTER("smurff_cg");
        iter2 = linop::solve_blockcg_impl(X2, K, reg, B, tol, blocksize, excess, throw_on_cholesky_error);
        SHOW(iter2);
        SHOW((X2 - B).norm());
    }

    SHOW((X2 - X1).norm());

    return iter1;
#endif
}

}}