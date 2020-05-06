#pragma once

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>
#include <Eigen/IterativeLinearSolvers>

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/counters.h>

#include <SmurffCpp/SideInfo/SparseSideInfo.h>

namespace smurff {
namespace linop {


//
//-- Solves the system (K' * K + reg * I) * X = B for X for m right-hand sides
//   K = d x n matrix
//   I = n x n identity
//   X = n x m matrix
//   B = n x m matrix
//
int solve_blockcg_1block(Matrix & X, const SparseSideInfo& K, double reg, Matrix & B, double tol, bool throw_on_cholesky_error = false);

/** good values for solve_blockcg are blocksize=32 an excess=8 */
int solve_blockcg(Matrix & X, const SparseSideInfo& K, double reg, Matrix & B, double tol, const int blocksize, const int excess, bool throw_on_cholesky_error = false);

int solve_blockcg_eigen(Matrix & X, const SparseSideInfo& K, double reg, Matrix & B, double tol, bool throw_on_cholesky_error = false);

}}