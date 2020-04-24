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
    387998143,910169995,2025160654,697139026,2038031161,1860560113,1004299869,2129921185,1039831747,1011153800,
  };

  testRandFunc(smurff::rand, "rand", expected_rand);
}


TEST_CASE("rand_normal", "[random]")
{
  const double expected_rand_normal[10] = {
    -1.26188,0.40815,-0.0157099,-1.56947,-1.1652,0.359155,1.34183,-1.16604,0.350437,0.373344,
  };

  testRandFunc(smurff::rand_normal, "rand_normal", expected_rand_normal);
}


TEST_CASE("rand_gamma", " [random]")
{
  const double expected_rand_gamma[10] = {
    0.465871,0.466284,0.463154,0.468155,0.462303,0.469327,0.466011,0.466946,0.465556,0.470684,
  };


  testRandFunc([]() -> double { return smurff::rand_gamma(35000.,0.00001333); }, "rand_gamma", expected_rand_gamma);
}
