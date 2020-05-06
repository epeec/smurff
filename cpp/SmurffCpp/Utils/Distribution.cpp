
// From:
// http://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
 
#include <limits>
#include <iostream>
#include <chrono>
#include <functional>
#include <random>

#ifdef EIGEN_USE_MKL_ALL
#include <mkl.h>
#endif

#include "SmurffCpp/Utils/counters.h"
#include "SmurffCpp/Utils/ThreadVector.hpp"
#include "SmurffCpp/Utils/omp_util.h"
#include "SmurffCpp/Utils/gamma_distribution.hpp"

#include <SmurffCpp/Types.h>

#include "Distribution.h"

namespace smurff {

typedef std::mt19937 rng;
typedef smurff::gamma_distribution<double> gamma_dist;
typedef std::uniform_real_distribution<double> uniform_dist;
typedef std::normal_distribution<double> normal_dist;

/*
 *  Init functions
 */
 
static thread_vector<rng> rngs;

#ifdef EIGEN_USE_MKL_ALL
static thread_vector<VSLStreamStatePtr> streams;
#endif

void init_bmrng() 
{
   using namespace std::chrono;
   auto ms = (duration_cast< milliseconds >(system_clock::now().time_since_epoch())).count();
   init_bmrng(ms);
}

void init_bmrng(int seed)
{
   std::vector<rng> r;
   for (int i = 0; i < threads::get_max_threads(); i++)
      r.push_back(rng(seed + i * 1999));
   rngs.init(r);

#ifdef EIGEN_USE_MKL_ALL
   std::vector<VSLStreamStatePtr> s(threads::get_max_threads());
   for (unsigned i = 0; i < s.size(); i++)
      vslNewStream(&s.at(i), VSL_BRNG_MT19937, (seed + i * 1999));
   streams.init(s);
#endif
}

template <typename Distribution>
double generate(Distribution &d)
{
   return d(rngs.local());
}

unsigned rand()
{
   return rngs.local()(); 
}

/*
 *  Normal distribution random numbers
 */
 
static void rand_normal(float_type* x, long n) 
{
   COUNTER("rand_normal");
#ifdef EIGEN_USE_MKL_ALL
   vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,streams.local(),n,x,0.0,1.0);
#else
   uniform_dist unif(-1.0, 1.0);

   for (long i = 0; i < n; i += 2) 
   {
      double x1, x2, w;

      do 
      {
         x1 = generate(unif);
         x2 = generate(unif);
         w = x1 * x1 + x2 * x2;
      } while ( w >= 1.0 );
 
      w = std::sqrt( (-2.0 * std::log( w ) ) / w );
      x[i] = x1 * w;

      if (i + 1 < n) 
      {
         x[i+1] = x2 * w;
      }
   }
#endif
}

double rand_normal() 
{
   float_type x;
   rand_normal(&x, 1);
   return x;
}

float_type RandNormalGenerator::operator()(float_type) const
{
   if ((c % 2) == 0) rand_normal(x, 2);
   return x[(c++)%2];
}

void rand_normal(Vector & x) 
{
   rand_normal(x.data(), x.size());
}
 
void rand_normal(Matrix & X) 
{
   rand_normal(X.data(), X.size());
}

double rand_unif(double low, double high) 
{
   uniform_dist unif(low, high);
   return generate(unif);
}

// returns random number according to Gamma distribution
// with the given shape (k) and scale (theta). See wiki.
double rand_gamma(double shape, double scale) 
{
   gamma_dist gamma(shape, scale);
   return generate(gamma);
}


//#define TEST_MVNORMAL

Matrix WishartUnit(int m, int df)
{
   Matrix c(m,m);
   c.setZero();

   for ( int i = 0; i < m; i++ ) 
   {
      gamma_dist gam(0.5*(df - i), 1.0);
      c(i,i) = std::sqrt(2.0 * generate(gam));
      c.block(i,i+1,1,m-i-1) = RandomVectorExpr(m-i-1);
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
   Matrix mu_o = MvNormal(Lam * kappa, mu);

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
Matrix MvNormal(const Matrix & Lambda, int num_samples)
{
   int ndims = Lambda.rows(); // Dimensionality 
   Eigen::LLT<Matrix> chol(Lambda);

   Matrix r(num_samples, ndims);
   rand_normal(r);
   Matrix ret = chol.matrixU().solve(r.transpose()).transpose();

#ifdef TEST_MVNORMAL
   std::cout << "MvNormal/2 {\n" << std::endl;
   std::cout << "  Lambda\n" << Lambda << std::endl;
   std::cout << "  nrows\n" << nrows << std::endl;
   std::cout << "  ret\n" << ret << std::endl;
   std::cout << "}\n" << std::endl;
#endif

   return ret;

}

Matrix MvNormal(const Matrix & Lambda, const Vector & mean, int num_samples)
{
   Matrix r = MvNormal(Lambda, num_samples);
   r.rowwise() += mean;
  
#ifdef TEST_MVNORMAL
   THROWERROR_ASSERT(r.rows() == nrows);
   std::cout << "MvNormal/2 {\n" << std::endl;
   std::cout << "  Lambda\n" << Lambda << std::endl;
   std::cout << "  mean\n" << mean << std::endl;
   std::cout << "  nrows\n" << nrows << std::endl;
   std::cout << "  r\n" << r << std::endl;
   std::cout << "}\n" << std::endl;
#endif

   return r;
}

} // end namespace smurff
