#include <stdexcept>
#include "heap_allocator.hpp"
#include "utility.hpp"

namespace ckt {
  const int32_t g_sub_bin_num = 8;

  const int32_t g_min_block_shift = 8;
  const int32_t g_max_block_shift = 26;

  const size_t g_min_block_bsize = (1<<g_min_block_shift);
  const size_t g_max_block_bsize = (1<<g_max_block_shift);
  const int32_t g_num_bins = g_max_block_shift - g_min_block_shift + 1;

  size_t g_current_alloc_bsize = 0;

  /*************************************************************/
  // Initialize a continuous region of memory divided by num_sub_bins,
  // each divided piece of memory has block_size
  bool SubBin::init(size_t block_size, int num_sub_bins) {
    m_block_size = block_size;
    m_bytes_this_sub_bin = block_size * num_sub_bins;
    m_num_sub_bins = num_sub_bins;
    char *base(nullptr);
    cudaMalloc((void**)(&base), m_bytes_this_sub_bin);

    check_cuda_error("cudaMalloc", __FILE__, __LINE__);
    if (base == nullptr) 
      return false; // out of memory

#ifdef DEBUG_HEAP_ALLOC
    g_current_alloc_bsize += m_bytes_this_sub_bin;
    float mega = (float)g_current_alloc_bsize/1000000.0;
    printf("cudaAllocing %p block_size %ld, bin size %ld global size %f MB\n", base, block_size, m_bytes_this_sub_bin, mega);
#endif      
    m_base.reset(base);
    m_used.resize(m_num_sub_bins);
    m_used_size.resize(m_num_sub_bins);
    m_used.assign(m_used.size(), false);
    m_used_count = 0;
    return true;
  }

  void *SubBin::request(const size_t &request_size) {
    if (m_block_size < request_size) {
      if (m_block_size > 0) 
        return nullptr;
      // This is a flexible sub-bin
      assert(m_num_sub_bins == 1);
      assert(is_full());
    }
      
    if (is_full()) {
      return nullptr;
    }
    for (size_t index = 0; index != m_used.size(); index++) {
      if (!m_used[index]) {
        m_used[index] = true;
        m_used_count++;
        m_used_size[index] = request_size;
        return (void *)(get_ptr() + index*m_block_size);
      }
    }
    return nullptr;
  }

  bool SubBin::remove(void *ptr) {
    if (is_in_range(ptr)) {
      size_t index = ((char*)ptr - get_ptr())/m_block_size;
      assert(m_used.size() > index);
      assert(m_used[index]);
      m_used[index] = false;
      m_used_size[index] = 0;
      m_used_count--;
      return true;
    }
    return false;
  }

  /*************************************************************/
  bool Bin::grow_sub_bin(const size_t &request_size) {
    SubBin *sub_bin = new SubBin();
    {
      assert(request_size <= m_block_size);
      if (!sub_bin->init(m_block_size, m_sub_bin_num))  {
        delete sub_bin;
        return false;
      }
    }
    m_bins.emplace_back(std::unique_ptr<SubBin>(sub_bin));
    return true;
  }
  
  void *Bin::allocate(const size_t &request_size) {
    if (request_size > m_block_size)
      return nullptr;

    void *ptr(nullptr);
    // try to get an empty slot from existing bins
    for (auto i = m_bins.begin(); i != m_bins.end(); i++) {
      ptr = (*i)->request(request_size);
      if (ptr != nullptr) 
        return ptr;
    }
    
    if (!grow_sub_bin(request_size)) 
      return nullptr;

    return m_bins.back()->request(request_size);
  }

  bool Bin::free(void *ptr) {
    for (auto i = m_bins.begin(); i != m_bins.end(); i++) {
      if ((*i)->remove(ptr)) {
        if ((*i)->is_empty()) {
          m_bins.erase(i);
        }
        return true;
      }
    }
    return false;
  }

  /*************************************************************/
  HeapAllocator::HeapAllocator(size_t min_block_size, size_t max_block_size, int32_t sub_bin_num):
    m_min_block_size(min_block_size),
    m_max_block_size(max_block_size) ,
    m_sub_bin_num(sub_bin_num)
  {
    m_num_bins = 0;
    for (size_t bsize = m_min_block_size; bsize <= m_max_block_size; bsize<<=1)
      {
        m_bins.push_back(Bin());
        m_bins.back().init(bsize, sub_bin_num);
        m_num_bins++;
      }
    m_bins.push_back(Bin());
    // the last bin has variable size, for big buffers
    m_bins.back().init(0, 1);
    m_num_bins++;
  }

  void *HeapAllocator::allocate(size_t bsize)
  {
    int bin_id = bin_index(bsize);
    assert(bin_id < (int)m_bins.size());
    void *ptr =  m_bins[bin_id].allocate(bsize);
    if (ptr == nullptr) {
      return ptr;
      /*
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        check_cuda_error("cudaMemGetInfo", __FILE__, __LINE__);	
        fprintf(stderr, "Error!!! Running out of GPU memory free %f MB, !!!!! Do defragmentating (TODO).\n",
          (float)free/1e6);
        abort();
      */
    }
    m_alloc_size += bsize;
    m_max_alloc_size = (std::max)(m_max_alloc_size, m_alloc_size);
    return ptr;
  }
  
  void HeapAllocator::deallocate(void *ptr, size_t bsize)
  {
    int bin_id = bin_index(bsize);
    assert(bin_id < (int)m_bins.size());
    m_bins[bin_id].free(ptr);
    m_alloc_size -= bsize;
  }

  int HeapAllocator::bin_index(const size_t bsize) 
  {
    if (bsize <= m_min_block_size) {
      return 0 ;
    }
    if (bsize > m_max_block_size) {
      return m_num_bins - 1;
    }
    size_t bit = m_min_block_size;
    int bin_id = 0;
    while (bit < bsize) {
      bit <<= 1;
      ++bin_id;
    }
    return bin_id;
  }
}
