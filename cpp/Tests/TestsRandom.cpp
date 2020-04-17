#include "catch.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include <limits>

#include <boost/version.hpp>

#include <SmurffCpp/Utils/Distribution.h>

TEST_CASE("Test random number generation - BOOST version", "[random]")
{
#if defined(USE_BOOST_RANDOM)
  static_assert((BOOST_VERSION / 1000) == EXPECTED_BOOST_SHORT_VERSION, "Wrong BOOST version");
  // Describes the boost version number in XYYYZZ format such that:
  // (BOOST_VERSION % 100) is the sub-minor version, ((BOOST_VERSION / 100) %
  // 1000) is the minor version, and (BOOST_VERSION / 100000) is the major
  // version.
  std::cout << "Using Boost "
            << BOOST_VERSION / 100000 << "."     // major version
            << BOOST_VERSION / 100 % 1000 << "." // minor version
            << BOOST_VERSION % 100               // patch level
            << std::endl;
#else
  WARN("Testing with std random - expect many failures\n");
#endif
}

#if defined(USE_BOOST_RANDOM)
template <typename Func>
void testRandFunc(const Func generator, const std::string &name, const double expected[10])
{
  std::vector<double> rnd(10);

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

TEST_CASE("Test rand_normal number generation", "[random]")
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

#endif
