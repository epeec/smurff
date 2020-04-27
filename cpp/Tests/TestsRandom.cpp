#include "catch.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include <limits>

#include <boost/version.hpp>

#include <SmurffCpp/Utils/Distribution.h>

static const int N = 10;

template <typename Func, typename T>
void testRandFunc(const Func generator, const std::string &name, const T expected[N])
{
  std::vector<T> rnd(N);

  smurff::init_bmrng(1234);

  for (int i = 0; i < N; i++) rnd[i] = generator();

#if 0
  std::cout << " // generated random: " << name << std::endl;
  for (int i = 0; i < N; i++) std::cout << rnd[i] << ",";
  std::cout << std::endl;
#endif

  for (int i = 0; i < N; i++)
    CHECK(rnd[i] == Approx(expected[i]).epsilon(smurff::approx_epsilon<smurff::float_type>()));
}


TEST_CASE("rand", "[random]")
{
  const std::uint64_t expected_rand[N] = {
    2468,1761648440,2085132941,2097745127,2097680755,360986329,575176667,1857859430,3274619740,3786690822,
  };

  testRandFunc(smurff::rand, "rand", expected_rand);
}

TEST_CASE("rand_unif", "[random]")
{
  const double expected_rand_unif[N] = {
    1.33791e-16,5.6116e-10,5.78697e-10,0.00470734,0.00941467,0.00954846,0.00498688,0.262682,0.230149,0.315679,
  };

  testRandFunc(
    []() -> double { return smurff::rand_unif(.0,1.); },
    "rand_unif", expected_rand_unif);
}


TEST_CASE("rand_normal", "[random]")
{
  const double expected_rand_normal[N] = {
    -1.07701,-0.924323,-0.726249,-2.3854,0.300049,0.56813,0.668291,-0.740228,-1.01351,-1.13575,
  };

  testRandFunc(smurff::rand_normal, "rand_normal", expected_rand_normal);
}


TEST_CASE("rand_gamma", " [random]")
{
  const double expected_rand_gamma[N] = {
    0.466537,0.466537,0.466641,0.466592,0.469649,0.469502,0.46723,0.465377,0.467955,0.465251,
  };


  testRandFunc([]() -> double { return smurff::rand_gamma(35000.,0.00001333); }, "rand_gamma", expected_rand_gamma);
}
