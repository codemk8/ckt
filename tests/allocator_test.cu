#include <iostream>
#include "./catch.hpp"
#include "ckt/include/heap_allocator.hpp"
using namespace ckt;

TEST_CASE( "SubBin", "simple" ) {
    SubBin sub_bin;
    int num_sub_bins = 8;
    int byte_size = (1<<10);
    auto success = sub_bin.init(byte_size, num_sub_bins);
    REQUIRE(success);

    // try request a larger size
    auto p = sub_bin.request(byte_size * 2);
    REQUIRE(p == nullptr);    

    // use all sub bins
    void *last_p = nullptr;
    for (int req = 0; req != num_sub_bins; req++){
        last_p = sub_bin.request(byte_size);
        REQUIRE(last_p != nullptr);
    }
    p = sub_bin.request(byte_size);
    REQUIRE(p == nullptr);

    // free the last one
    success = sub_bin.remove(last_p);
    REQUIRE(success);

    // the next allocate should return last_p
    auto new_p = sub_bin.request(byte_size);
    REQUIRE(new_p == last_p);
}

TEST_CASE( "Bin", "simple" ) {
    Bin bin;
    int num_sub_bins = 8;
    int byte_size = (1<<10);
    bin.init(byte_size, num_sub_bins);
    
    // Requesting a larger size than expected should return nullptr
    auto p = bin.allocate(byte_size * 2);
    REQUIRE(p == nullptr);    

    // we can request more subbins now with bins
    void *last_p = nullptr;
    for (int req = 0; req != 2 * num_sub_bins; req++){
        last_p = bin.allocate(byte_size);
        REQUIRE(last_p != nullptr);
    }

    // free the last one
    auto success = bin.free(last_p);
    REQUIRE(success);
}

TEST_CASE("HeapAllocator", "simple") {
    HeapAllocator ha(1<<6, 1<<15, 8);

    int index = ha.bin_index(1<<6);
    REQUIRE(index == 0);
    index = ha.bin_index(1<<5);
    REQUIRE(index == 0);

    index = ha.bin_index((1<<6) + 1);
    REQUIRE(index == 1);

    index = ha.bin_index((1<<15));
    REQUIRE(index == (15-6));

     index = ha.bin_index((1<<15) + 1);
    REQUIRE(index == (15-6+1));
}

TEST_CASE("HeapAllocator_allocate", "walkthrough") {
    HeapAllocator ha(1<<6, 1<<15, 8);

    for (int r = 0; r != 100; ++r) {
        int rsize = rand() % (1<<15);
        auto p = ha.allocate(rsize);
        REQUIRE(p != 0);
    }
    // std::this_thread::sleep_for(std::chrono::seconds(10));    
}