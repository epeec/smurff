
// From:
// http://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
 
#include <iostream>
#include <chrono>
#include <functional>

#include "Utils/ThreadVector.hpp"
#include "Utils/omp_util.h"

#ifdef USE_BOOST_RANDOM
#include <boost/random.hpp>
#define MERSENNE_TWISTER boost::random::mt19937
#define UNIFORM_REAL_DISTRIBUTION boost::random::uniform_real_distribution<double>
#define GAMMA_DISTRIBUTION boost::random::gamma_distribution<double>
#else
#include <random>
#define MERSENNE_TWISTER std::mt19937
#define UNIFORM_REAL_DISTRIBUTION std::uniform_real_distribution<double>
#define GAMMA_DISTRIBUTION std::gamma_distribution<double>
#endif

#include <SmurffCpp/Types.h>

#include "Distribution.h"

namespace smurff {

static thread_vector<MERSENNE_TWISTER> bmrngs;

double randn0()
{
   return bmrandn_single_thread();
}

double randn(double) 
{
   return bmrandn_single_thread();
}

void bmrandn(float_type* x, long n) 
{
   #pragma omp parallel 
   {
      UNIFORM_REAL_DISTRIBUTION unif(-1.0, 1.0);
      auto& bmrng = bmrngs.local();
      
      #pragma omp for schedule(static)
      for (long i = 0; i < n; i += 2) 
      {
         double x1, x2, w;
         do 
         {
           x1 = unif(bmrng);
           x2 = unif(bmrng);
           w = x1 * x1 + x2 * x2;
         } while ( w >= 1.0 );
   
         w = std::sqrt( (-2.0 * std::log( w ) ) / w );
         x[i] = x1 * w;

         if (i + 1 < n) 
         {
           x[i+1] = x2 * w;
         }
      }
   }
}
   
void bmrandn(Matrix & X) 
{
   long n = X.rows() * (long)X.cols();
   bmrandn(X.data(), n);
}

double bmrandn_single_thread() 
{
   //TODO: add bmrng as input
   UNIFORM_REAL_DISTRIBUTION unif(-1.0, 1.0);
   auto& bmrng = bmrngs.local();
  
   double x1, x2, w;
   do 
   {
      x1 = unif(bmrng);
      x2 = unif(bmrng);
      w = x1 * x1 + x2 * x2;
   } while ( w >= 1.0 );

   w = std::sqrt( (-2.0 * std::log( w ) ) / w );
   return x1 * w;
}

// to be called within OpenMP parallel loop (also from serial code is fine)
void bmrandn_single_thread(float_type* x, long n) 
{
   UNIFORM_REAL_DISTRIBUTION unif(-1.0, 1.0);
   auto& bmrng = bmrngs.local();

   for (long i = 0; i < n; i += 2) 
   {
      double x1, x2, w;

      do 
      {
         x1 = unif(bmrng);
         x2 = unif(bmrng);
         w = x1 * x1 + x2 * x2;
      } while ( w >= 1.0 );
 
      w = std::sqrt( (-2.0 * std::log( w ) ) / w );
      x[i] = x1 * w;

      if (i + 1 < n) 
      {
         x[i+1] = x2 * w;
      }
   }
}
  
void bmrandn_single_thread(Vector & x) 
{
   bmrandn_single_thread(x.data(), x.size());
}
 
void bmrandn_single_thread(Matrix & X) 
{
   long n = X.rows() * (long)X.cols();
   bmrandn_single_thread(X.data(), n);
}


void init_bmrng() 
{
   using namespace std::chrono;
   auto ms = (duration_cast< milliseconds >(system_clock::now().time_since_epoch())).count();
   init_bmrng(ms);
}

void init_bmrng(int seed) 
{
    std::vector<MERSENNE_TWISTER> v;
    for (int i = 0; i < threads::get_max_threads(); i++)
    {
        v.push_back(MERSENNE_TWISTER(seed + i * 1999));
    }

    bmrngs.init(v);
}
   
double rand_unif() 
{
   UNIFORM_REAL_DISTRIBUTION unif(0.0, 1.0);
   auto& bmrng = bmrngs.local();
   return unif(bmrng);
}
 
double rand_unif(double low, double high) 
{
   UNIFORM_REAL_DISTRIBUTION unif(low, high);
   auto& bmrng = bmrngs.local();
   return unif(bmrng);
}

// returns random number according to Gamma distribution
// with the given shape (k) and scale (theta). See wiki.
double rgamma(double shape, double scale) 
{
   GAMMA_DISTRIBUTION gamma(shape, scale);
   return gamma(bmrngs.local());
}

auto nrandn(int n) -> decltype(Vector::NullaryExpr(n, std::cref(randn))) 
{
   return Vector::NullaryExpr(n, std::cref(randn));
}

auto nrandn(int n, int m) -> decltype(Array2D::NullaryExpr(n, m, std::cref(randn)))
{
   return Array2D::NullaryExpr(n, m, std::cref(randn)); 
}

//#define TEST_MVNORMAL

Matrix WishartUnit(int m, int df)
{
   Matrix c(m,m);
   c.setZero();
   auto& rng = bmrngs.local();

   for ( int i = 0; i < m; i++ ) 
   {
      GAMMA_DISTRIBUTION gam(0.5*(df - i));
      c(i,i) = std::sqrt(2.0 * gam(rng));
      Vector r = nrandn(m-i-1);
      c.block(i,i+1,1,m-i-1) = r;
   }

   Matrix ret = c.transpose() * c;

   #ifdef TEST_MVNORMAL
   std::cout << "WISHART UNIT {\n" << std::endl;
   std::cout << "  m:\n" << m << std::endl;
   std::cout << "  df:\n" << df << std::endl;
   std::cout << "  ret;\n" << ret << std::endl;
   std::cout << "  c:\n" << c << std::endl;
   std::cout << "}\n" << std::endl;
   #endif

   return ret;
}

Matrix Wishart(const Matrix &sigma, const int df)
{
   //  Get R, the upper triangular Cholesky factor of SIGMA.
   auto chol = sigma.llt();
   Matrix r = chol.matrixL();

   //  Get AU, a sample from the unit Wishart distribution.
   Matrix au = WishartUnit(sigma.rows(), df);

   //  Construct the matrix A = R' * AU * R.
   Matrix a = r * au * chol.matrixU();

   #ifdef TEST_MVNORMAL
   std::cout << "WISHART {\n" << std::endl;
   std::cout << "  sigma:\n" << sigma << std::endl;
   std::cout << "  r:\n" << r << std::endl;
   std::cout << "  au:\n" << au << std::endl;
   std::cout << "  df:\n" << df << std::endl;
   std::cout << "  a:\n" << a << std::endl;
   std::cout << "}\n" << std::endl;
   #endif

  return a;
}

// from julia package Distributions: conjugates/normalwishart.jl
std::pair<Vector, Matrix> NormalWishart(const Vector & mu, double kappa, const Matrix & T, const int nu)
{
   Matrix Lam = Wishart(T, nu);
   Matrix mu_o = MvNormal_prec(Lam * kappa, mu);

   #ifdef TEST_MVNORMAL
   std::cout << "NORMAL WISHART {\n" << std::endl;
   std::cout << "  mu:\n" << mu << std::endl;
   std::cout << "  kappa:\n" << kappa << std::endl;
   std::cout << "  T:\n" << T << std::endl;
   std::cout << "  nu:\n" << nu << std::endl;
   std::cout << "  mu_o\n" << mu_o << std::endl;
   std::cout << "  Lam\n" << Lam << std::endl;
   std::cout << "}\n" << std::endl;
   #endif

   return std::make_pair(mu_o , Lam);
}

std::pair<Vector, Matrix> CondNormalWishart(const int N, const Matrix &NS, const Vector &NU, const Vector &mu, const double kappa, const Matrix &T, const int nu)
{
   int nu_c = nu + N;

   double kappa_c = kappa + N;
   auto mu_c = (kappa * mu + NU) / (kappa + N);
   auto X    = T + NS + kappa * mu.transpose() * mu - kappa_c * mu_c.transpose() * mu_c;
   Matrix T_c = X.inverse();
    
   const auto ret = NormalWishart(mu_c, kappa_c, T_c, nu_c);

#ifdef TEST_MVNORMAL
   std::cout << "CondNormalWishart/7 {\n" << std::endl;
   std::cout << "  mu:\n" << mu << std::endl;
   std::cout << "  kappa:\n" << kappa << std::endl;
   std::cout << "  T:\n" << T << std::endl;
   std::cout << "  nu:\n" << nu << std::endl;
   std::cout << "  N:\n" << N << std::endl;
   std::cout << "  NS:\n" << NS << std::endl;
   std::cout << "  NU:\n" << NU << std::endl;
   std::cout << "  mu_o\n" << ret.first << std::endl;
   std::cout << "  Lam\n" << ret.second << std::endl;
   std::cout << "}\n" << std::endl;
#endif

   return ret;
}

std::pair<Vector, Matrix> CondNormalWishart(const Matrix &U, const Vector &mu, const double kappa, const Matrix &T, const int nu)
{
   auto N = U.rows();
   auto NS = U.transpose() * U;
   auto NU = U.colwise().sum();

#ifdef TEST_MVNORMAL
   std::cout << "CondNormalWishart/5 {\n" << std::endl;
   std::cout << "  U:\n" << U << std::endl;
   std::cout << "}\n" << std::endl;
#endif

   return CondNormalWishart(N, NS, NU, mu, kappa, T, nu);
}

// Normal(0, Lambda^-1) for nn columns
Matrix MvNormal_prec(const Matrix & Lambda, int nrows)
{
   int ncols = Lambda.rows(); // Dimensionality (rows)
   Eigen::LLT<Matrix> chol(Lambda);

   Matrix r(nrows, ncols);
   bmrandn(r);
   Matrix ret = chol.matrixU().solve(r.transpose()).transpose();

#ifdef TEST_MVNORMAL
   std::cout << "MvNormal_prec/2 {\n" << std::endl;
   std::cout << "  Lambda\n" << Lambda << std::endl;
   std::cout << "  nrows\n" << nrows << std::endl;
   std::cout << "  ret\n" << ret << std::endl;
   std::cout << "}\n" << std::endl;
#endif

   return ret;

}

Matrix MvNormal_prec(const Matrix & Lambda, const Vector & mean, int nrows)
{
   Matrix r = MvNormal_prec(Lambda, nrows);
   r.rowwise() += mean;
  
#ifdef TEST_MVNORMAL
   THROWERROR_ASSERT(r.rows() == nrows);
   std::cout << "MvNormal_prec/2 {\n" << std::endl;
   std::cout << "  Lambda\n" << Lambda << std::endl;
   std::cout << "  mean\n" << mean << std::endl;
   std::cout << "  nrows\n" << nrows << std::endl;
   std::cout << "  r\n" << r << std::endl;
   std::cout << "}\n" << std::endl;
#endif

   return r;
}

// Draw n samples from a dim-dimensional normal distribution
// with a specified mean and covariance
Matrix MvNormal(const Matrix &covar, const Vector &mean, int num_samples) 
{
   THROWERROR_ASSERT(mean.nonZeros() == covar.rows());
   THROWERROR_ASSERT(mean.nonZeros() == covar.cols());
   auto dim = mean.nonZeros();
   auto normSamples = Matrix::NullaryExpr(num_samples, dim, std::cref(randn));
   return covar.llt().solve(normSamples.transpose()).transpose().rowwise() + mean;
}

} // end namespace smurff
