#define CATCH_CONFIG_MAIN  

#include "./catch.hpp"
#include "ckt/include/cuda_config.hpp"
#include "ckt/include/vector.hpp"

using namespace ckt;

TEST_CASE( "ckt_test", "[CktArray simple]" ) {
    CktArray<int> ints(10);
    REQUIRE(ints.size() == 10);
}