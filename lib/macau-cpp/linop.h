#ifndef LINOP_H
#define LINOP_H

#include <Eigen/Dense>

extern "C" {
  #include <csr.h>
}

class SparseFeat {
  public:
    BinaryCSR M;
    BinaryCSR Mt;

    SparseFeat() {}

    SparseFeat(int nrow, int ncol, long nnz, int* rows, int* cols) {
      M  = * new_bcsr(nnz, nrow, ncol, rows, cols);
      Mt = * new_bcsr(nnz, ncol, nrow, cols, rows);
    }
    int nfeat()    {return M.ncol;}
    int nsamples() {return M.nrow;}
};

void At_mul_A( Eigen::MatrixXd & out, const SparseFeat & A);

void A_mul_B(  Eigen::VectorXd & out, BinaryCSR & csr, const Eigen::VectorXd & b);
void A_mul_Bt( Eigen::MatrixXd & out, BinaryCSR & csr, const Eigen::MatrixXd & B);
int  solve(    Eigen::MatrixXd &   X, SparseFeat & sparseFeat, double reg, Eigen::MatrixXd & B, double tol);

// some util functions:
void At_mul_A_blas(const Eigen::MatrixXd & A, double* AtA);
void A_mul_At_blas(const Eigen::MatrixXd & A, double* AAt);
void A_mul_B_blas(Eigen::MatrixXd & Y, const Eigen::MatrixXd & A, const Eigen::MatrixXd & B);
void A_mul_Bt_blas(Eigen::MatrixXd & Y, const Eigen::MatrixXd & A, const Eigen::MatrixXd & B);

// Y = beta * Y + alpha * A * B (where B is symmetric)
void Asym_mul_B_left(double beta, Eigen::MatrixXd & Y, double alpha, Eigen::MatrixXd & A, Eigen::MatrixXd & B);
void Asym_mul_B_right(double beta, Eigen::MatrixXd & Y, double alpha, Eigen::MatrixXd & A, Eigen::MatrixXd & B);

#endif /* LINOP_H */
