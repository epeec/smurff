#include "catch.hpp"

#include <SmurffCpp/Types.h>

#include <Utils/Error.h>

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/Distribution.h>

namespace smurff {

namespace mu = smurff::matrix_utils;

TEST_CASE( "mvnormal" ) {
  init_bmrng(1234);

  const int num_samples = 1000;
  Vector mean = mu::make_dense({10, 1} , { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.});
  Matrix covar = Matrix::Identity(10,10);

  auto randomMatrix = MvNormal(covar, mean, num_samples);

  // check mean
  REQUIRE(mu::equals_vector(randomMatrix.rowwise().sum(), num_samples * mean, num_samples/10));

  // check variance
  Matrix centered = (randomMatrix.colwise() - mean);
  Matrix squared = centered.array().square(); 
  Vector var = squared.rowwise().sum() / num_samples;
  REQUIRE(mu::equals_vector(var, covar.diagonal(), 0.1));

}


TEST_CASE( "mvnormal/prec" ) {
  init_bmrng(1234);

  const int num_samples = 1000;
  Vector mean = mu::make_dense({10, 1} , { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.});
  Matrix covar = Matrix::Identity(10,10);

  auto randomMatrix = MvNormal(covar, mean, num_samples);

  // check mean
  REQUIRE(mu::equals_vector(randomMatrix.rowwise().sum(), num_samples * mean, num_samples/10));

  // check variance
  Matrix centered = (randomMatrix.colwise() - mean);
  Matrix squared = centered.array().square(); 
  Vector var = squared.rowwise().sum() / num_samples;
  REQUIRE(mu::equals_vector(var, covar.diagonal(), 0.1));

}


}