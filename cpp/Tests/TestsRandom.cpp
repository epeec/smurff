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
  const std::uint64_t expected_rand[10] = {
    2468,1761648440,2085132941,2097745127,2097680755,360986329,575176667,1857859430,3274619740,3786690822,
  };

  testRandFunc(smurff::rand, "rand", expected_rand);
}


TEST_CASE("rand_normal", "[random]")
{
  const double expected_rand_normal[10] = {
    -1.07701,-0.924323,-0.726249,-2.3854,0.300049,0.56813,0.668291,-0.740228,-1.01351,-1.13575,
  };

  testRandFunc(smurff::rand_normal, "rand_normal", expected_rand_normal);
}


TEST_CASE("rand_gamma", " [random]")
{
  const double expected_rand_gamma[10] = {
    0.463768,0.463655,0.465961,0.46779,0.467365,0.47225,0.461538,0.465755,0.471875,0.463198,
  };


  testRandFunc([]() -> double { return smurff::rand_gamma(35000.,0.00001333); }, "rand_gamma", expected_rand_gamma);
}
