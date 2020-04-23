#include "catch.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include <limits>

#include <boost/version.hpp>

#include <SmurffCpp/Utils/Distribution.h>

template <typename Func, typename T>
void testRandFunc(const Func generator, const std::string &name, const T expected[10])
{
  std::vector<T> rnd(10);

  smurff::init_bmrng(1234);

  for (int i = 0; i < 10; i++) rnd[i] = generator();

#if 0
  std::cout << " // generated random: " << name << std::endl;
  for (int i = 0; i < 10; i++) std::cout << rnd[i] << ",";
  std::cout << std::endl;
#endif

  for (int i = 0; i < 10; i++)
    CHECK(rnd[i] == Approx(expected[i]).epsilon(APPROX_EPSILON));
}


TEST_CASE("rand", "[random]")
{
  const unsigned expected_rand[10] = {
      1812433295,
      3624866548,
      1142332505,
      2954765758,
      472231715,
      2284664968,
      4097098221,
      1614564178,
      3426997431,
      944463388,
  };

  testRandFunc(smurff::rand, "rand", expected_rand);
}


TEST_CASE("rand_normal", "[random]")
{
  const double expected_rand_normal[10] = {
#include "TestsRandom_ExpectedRandNormal.h"
  };

  testRandFunc(smurff::rand_normal, "rand_normal", expected_rand_normal);
}


TEST_CASE("rand_gamma", " [random]")
{
  const double expected_rand_gamma[10] = {
#include "TestsRandom_ExpectedRandGamma.h"
  };


  testRandFunc([]() -> double { return smurff::rand_gamma(1.,2.); }, "rand_gamma", expected_rand_gamma);
}
