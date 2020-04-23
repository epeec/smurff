#pragma once

#include <map>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>

namespace smurff
{
   
   void init_bmrng();
   void init_bmrng(int seed);

   // Generate single numbers 

   double rand_normal();
   double rand_unif(double low = 0.0, double high = 1.);
   double rand_gamma(double shape, double scale);

   // Generate stream of random numbers

   struct RandNormalGenerator
   {
      mutable unsigned c = 0;
      mutable float_type x[2];

      float_type operator()(float_type) const;
   };
  
#define RandomVectorExpr(n) (Vector::NullaryExpr(n, RandNormalGenerator())) 

   // Wishart distribution
   
   std::pair<Vector, Matrix> NormalWishart(const Vector & mu, double kappa, const Matrix & T, double nu);
   std::pair<Vector, Matrix> CondNormalWishart(const Matrix &U, const Vector &mu, const double kappa, const Matrix &T, const int nu);
   std::pair<Vector, Matrix> CondNormalWishart(const int N, const Matrix &NS, const Vector &NU, const Vector &mu, const double kappa, const Matrix &T, const int nu);
   
   // Multivariate normal gaussian

   Matrix MvNormal(const Matrix & Lambda, int nn = 1);
   Matrix MvNormal(const Matrix & Lambda, const Vector & mean, int nn = 1);
}
