#pragma once

#include <map>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>

namespace smurff
{
   double randn0();
   double randn(double = .0);
   
   void bmrandn(float_type* x, long n);
   void bmrandn(Matrix & X);
   
   double bmrandn_single_thread();
   void bmrandn_single_thread(float_type* x, long n);
   void bmrandn_single_thread(Vector & x);
   void bmrandn_single_thread(Matrix & X);
   
   void init_bmrng();
   void init_bmrng(int seed);
   
   double rand_unif();
   double rand_unif(double low, double high);
   
   double rgamma(double shape, double scale);
   
   // return a random matrix of size n, m
   
   auto nrandn(int n) -> decltype(Vector::NullaryExpr(n, std::cref(randn)) ); 
   auto nrandn(int n, int m) -> decltype(Array2D::NullaryExpr(n, m, std::cref(randn)) );
   
   // Wishart distribution
   
   std::pair<Vector, Matrix> NormalWishart(const Vector & mu, double kappa, const Matrix & T, double nu);
   std::pair<Vector, Matrix> CondNormalWishart(const Matrix &U, const Vector &mu, const double kappa, const Matrix &T, const int nu);
   std::pair<Vector, Matrix> CondNormalWishart(const int N, const Matrix &NS, const Vector &NU, const Vector &mu, const double kappa, const Matrix &T, const int nu);
   
   // Multivariate normal gaussian

   Matrix MvNormal_prec(const Matrix & Lambda, int nn = 1);
   Matrix MvNormal_prec(const Matrix & Lambda, const Vector & mean, int nn = 1);
   Matrix MvNormal(const Matrix &covar, const Vector &mean, int nn = 1);
}
