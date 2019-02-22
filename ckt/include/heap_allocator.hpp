#pragma once
#include <cassert>
#include <cstdio>
#include <memory>
#include <vector>
#include <list>
#include <cuda_runtime_api.h>
#include "allocator.hpp"

namespace ckt {
  extern const int32_t g_sub_bin_num;
  extern const size_t g_min_block_bsize;
  extern const size_t g_max_block_bsize;
  extern const int32_t g_num_bins;

  // create bins with different sizes. Size is two to the power of N, where N is [MIN_BLOCK_SHIFT, MAX_BLOCK_SHIFT]
  // each bin will grow by SUB_BIN_SIZE each time it runs out of entries
  // for memory request larger than 2^MAX_BLOCK_SHIFT, track them one by one (SUB_BIN_SIZE is 1), in the last bin
#define DEBUG_HEAP_ALLOC
#undef DEBUG_HEAP_ALLOC

// extern std::atomic_int64_t g_total_alloc_bsizes;

extern  size_t g_current_alloc_bsizes;

  template <class T>
  struct cuda_heap_deleter {
    void operator () (T *ptr) const {   
      cudaFree((void *)ptr);
    }
  };


  /*
   A SubBin maintains a fixed number of memory blocks in a continuous region
   Once initialized, it cannot grow in size
  */
  class SubBin {
  public:
#ifdef DEBUG
    ~SubBin() {
      g_current_alloc_bsize -= m_bytes_this_sub_bin;
      assert(is_empty());
    }
#endif
    bool init(size_t block_size, int BIN_SIZE);
    void *request(const size_t &request_size);
    bool remove(void *ptr); 

    bool is_empty() const {
      return (m_used_count == 0);
    }

    bool is_in_range(void *ptr) {
      if (get_ptr() == NULL)
        return false;
      if (ptr >= get_ptr() && ptr < get_ptr() + m_block_size * bin_size())
        return true;
      return false;
    }
  private:
    bool is_full() const {
      return (m_used_count == m_used.size());
    }

    size_t count_used() const {
      return m_used_count;
    }
    
    size_t bin_size() const { return m_used.size(); }
    char *get_ptr() { return m_base.get(); }


    size_t m_block_size  = 0;
    std::unique_ptr<char, cuda_heap_deleter<char>> m_base;
    std::vector<bool> m_used;
    std::vector<size_t> m_used_size;
    size_t m_used_count = 0;
    size_t m_bytes_this_sub_bin = 0;

  };

  /* 
   * A Bin manages a list of SubBin to try the best to allocate memory blocks
   */
  class Bin {
  public:
    void init(size_t block_size, int sub_bin_num) {
      m_block_size = block_size;
      m_sub_bin_num = sub_bin_num;
    }

    void *allocate(const size_t &request_size);
    bool free(void *ptr);
  private:
    bool grow_sub_bin(const size_t &request_size);
    size_t num_sub_bins() const { return m_bins.size(); }

    size_t m_block_size = 0;
    int m_sub_bin_num = 1;
    std::list<std::unique_ptr<SubBin>> m_bins;
  };


  class HeapAllocator : public GpuAllocator {
public:
    HeapAllocator(size_t min_block_size = g_min_block_bsize, size_t max_block_size = g_max_block_bsize, int sub_bin_num = g_sub_bin_num);

    ~HeapAllocator() {
#ifdef DEBUG_HEAP_ALLOC
      float mega = (float)m_max_alloc_size/1000000.0;
      printf("HeapAllocator maximal allocated size %f\n", mega);
#endif
  }

    void *allocate(size_t bsize);
    void deallocate(void *ptr);

  // returns the nearest bin index according the required byte size
    int bin_index(const size_t bsize);

private:
    std::vector<Bin> m_bins;

    size_t m_alloc_size = 0;
    size_t m_max_alloc_size = 0;

    size_t m_min_block_size = g_min_block_bsize;
    size_t m_max_block_size = g_max_block_bsize;
    int m_num_bins = g_num_bins;
    int m_sub_bin_num = g_sub_bin_num;
  };

}
