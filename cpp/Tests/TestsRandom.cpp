#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include <limits>

#include <boost/version.hpp>

TEST_CASE("Test random number generation", "[random]")
{
#if defined(USE_BOOST_RANDOM)
  static_assert ( (BOOST_VERSION / 1000) == EXPECTED_BOOST_SHORT_VERSION, "Wrong BOOST version");
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

#if defined(USE_BOOST_RANDOM) 
   init_bmrng(1234);

   double rnd = 0.0;
   rnd = rand_normal();
   REQUIRE(rnd == Approx(-1.38981).epsilon(APPROX_EPSILON));

   rnd = rand_normal();
   REQUIRE(rnd == Approx(0.444601).epsilon(APPROX_EPSILON));

   rnd = rand_normal();
   REQUIRE(rnd == Approx(-1.13281).epsilon(APPROX_EPSILON));

   rnd = rand_normal();
   REQUIRE(rnd == Approx(0.708248).epsilon(APPROX_EPSILON));

   rnd = rand_normal();
   REQUIRE(rnd == Approx(0.369621).epsilon(APPROX_EPSILON));

   rnd = rand_normal();
   REQUIRE(rnd == Approx(-0.465294).epsilon(APPROX_EPSILON));

   rnd = rand_normal();
   REQUIRE(rnd == Approx(-0.637987).epsilon(APPROX_EPSILON));

   rnd = rand_normal();
   REQUIRE(rnd == Approx(0.510229).epsilon(APPROX_EPSILON));

   rnd = rand_normal();
   REQUIRE(rnd == Approx(0.28734).epsilon(APPROX_EPSILON));

   rnd = rand_normal();
   REQUIRE(rnd == Approx(1.22677).epsilon(APPROX_EPSILON));
   #endif
}

TEST_CASE("rand_gamma", " [random]")
{
   #ifdef USE_BOOST_RANDOM
   init_bmrng(1234);

   double rnd = 0.0;
   rnd = rand_gamma(1, 2);
   REQUIRE(rnd == Approx(0.425197).epsilon(APPROX_EPSILON));

   rnd = rand_gamma(1, 2);
   REQUIRE(rnd == Approx(1.37697).epsilon(APPROX_EPSILON));

   rnd = rand_gamma(1, 2);
   REQUIRE(rnd == Approx(1.9463).epsilon(APPROX_EPSILON));

   rnd = rand_gamma(1, 2);
   REQUIRE(rnd == Approx(3.40572).epsilon(APPROX_EPSILON));

   rnd = rand_gamma(1, 2);
   REQUIRE(rnd == Approx(1.15154).epsilon(APPROX_EPSILON));

   rnd = rand_gamma(1, 2);
   REQUIRE(rnd == Approx(1.89408).epsilon(APPROX_EPSILON));

   rnd = rand_gamma(1, 2);
   REQUIRE(rnd == Approx(3.07757).epsilon(APPROX_EPSILON));

   rnd = rand_gamma(1, 2);
   REQUIRE(rnd == Approx(2.95121).epsilon(APPROX_EPSILON));

   rnd = rand_gamma(1, 2);
   REQUIRE(rnd == Approx(3.02804).epsilon(APPROX_EPSILON));

   rnd = rand_gamma(1, 2);
   REQUIRE(rnd == Approx(3.94182).epsilon(APPROX_EPSILON));
#endif  
}
