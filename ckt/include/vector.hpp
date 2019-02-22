#pragma once

#include <cuda.h>
#include <iostream>
#include <fstream>
#include <string>
//#include "CudppPlanFactory.h"

#include <thrust/device_ptr.h>
//#include "SegmentArrayKernel.h"
#include <curand.h>
#include "./utility.hpp"
#include "./heap_manager.hpp"
#define USE_SHARED_PTR 1
#include <cstddef>

//extern cudaDeviceProp gDevProp;


namespace ckt {
  extern HeapManager gHeapManager;

  enum class ArrayType
  { 
    CPU_ARRAY = 0,
    GPU_ARRAY = 1
  };

  template <class T>
  class CktArray {
  public:
    // light-weighted initialization, lazy-initialization
    explicit CktArray(int size = 0):
        m_size(size),
        m_capacity(0),
        m_host_base(),
        m_dvce_base(),
        m_is_cpu_valid(false),
        m_is_gpu_valid(false),
        m_gpu_allocated(false),
        m_cpu_allocated(false),
        gpuNeedToFree(true),
        cpuNeedToFree(true)
    {
    }

    ~CktArray() {
      destroy();
    }
    
    // Copy Constructor
    CktArray(const CktArray<T> &rhs) 
    {
      init_state();
      deep_copy(rhs);
    }
    
    // Init from a std vector
    explicit CktArray(const std::vector<T> &rhs)
    {
      init_state();
      clean_resize(rhs.size());
      cudaMemcpy(getWOGpuPtr(), rhs.data(), size()*sizeof(T), cudaMemcpyDefault);
    }

    // Assign from a std vector
    void operator=(const std::vector<T> &rhs)
    {
      clean_resize(rhs.size());
      cudaMemcpy(getWOGpuPtr(), rhs.data(), size()*sizeof(T), cudaMemcpyDefault);
    }

    /* Init from raw pointers
      */
    void init(const T *ptr, size_t _size) {
      clean_resize(_size);
      T *g_ptr = getWOGpuPtr();
      cudaMemcpy(g_ptr, ptr, sizeof(T) * _size, cudaMemcpyDefault);
    }
    
    CktArray<T> &operator=(const CktArray<T> &rhs)
      {
        deep_copy(rhs);
        return *this;
      }


     bool GpuHasLatest() const {
      return m_is_gpu_valid;
    }

    bool CpuHasLatest() const {
      return m_is_cpu_valid;
    }
    
    // deep copy to
    void clone(CktArray<T> &dst) const
    {
      dst.clean_resize(size());
      if (m_is_gpu_valid)
        {
          cudaMemcpy(dst.getGpuPtr(), getROGpuPtr(), sizeof(T)*m_size, cudaMemcpyDeviceToDevice);
        }
      if (m_is_cpu_valid)
        {
          memcpy(dst.getPtr(), getROPtr(), sizeof(T)*m_size);
        }
      
      dst.m_is_cpu_valid = m_is_cpu_valid;
      dst.m_is_gpu_valid = m_is_gpu_valid;
      dst.m_gpu_allocated = m_gpu_allocated;
      dst.m_cpu_allocated = m_cpu_allocated;
    }

    void alias(const CktArray<T> & dst) {
      shallow_copy(dst);
      m_capacity = -1; // to disable calling free
    }
    
    void clear()
    {
      Resize(0);
    }

    int size() const
    {
      return m_size;
    }
    
    /*! A clean version of resize.
      It does not copy the old data, 
      nor does it initialize the data 
    */
    void clean_resize(int64_t _size) {
      /* TODO Reallocate if too waste of space */
      if (m_capacity >= _size || _size == 0)  {
        m_size = _size;
        m_is_gpu_valid = false;
        m_is_cpu_valid = false;
        return;
      }
      if (_size > 0)  {
        // depending on previous state
        if (m_gpu_allocated) {
          if (m_dvce_base)
            gHeapManager.Free(GPU_HEAP, m_dvce_base);
          m_dvce_base = gHeapManager.Malloc(GPU_HEAP, _size * sizeof(T));
          m_is_gpu_valid = false;
        }
        if (m_cpu_allocated) {
          if (m_host_base)
            gHeapManager.Free(CPU_HEAP, m_host_base);
          m_host_base = gHeapManager.Malloc(CPU_HEAP, _size * sizeof(T));
          m_is_cpu_valid = false;
        }
        m_size = _size;
        m_capacity = _size;
      }
    }

    void Resize(int64_t new_size) 
    {
      if (new_size <= m_capacity)
      {
        // free memory if resize to zero
        if (new_size == 0 && m_size > 0) {
          destroy();
        }
        m_size = new_size;
      }
      else // need to reallocate
      {    
        if (!m_gpu_allocated && !m_cpu_allocated) {
          m_size = new_size;
        }
        else  {
          if (m_gpu_allocated)
          {
              std::shared_ptr<T> newm_dvce_base((T*)GpuDeviceAllocator(new_size*sizeof(T)), GpuDeviceDeleter);
              if (m_is_gpu_valid)
                {
                  cudaError_t error = cudaMemcpy(newm_dvce_base, m_dvce_base, m_size*sizeof(T), cudaMemcpyDeviceToDevice);
                  assert(error == cudaSuccess);
                }
              m_dvce_base = newm_dvce_base;
          }
          if (m_cpu_allocated)
          {
              std::shared_ptr<T> newm_host_base((T*)GpuHostAllocator(new_size*sizeof(T)), GpuHostDeleter);
              if (m_is_cpu_valid)
                memcpy(newm_host_base, m_host_base, m_size*sizeof(T));
              m_host_base = newm_host_base;            
          }
        }
        m_size = new_size;
        m_capacity = new_size;
      }
    }

    /*! return the pointer without changing the internal state */
    T *getRawPtr() 
    {
      allocateCpuIfNecessary();
      return m_host_base;
    }

    /*! return the gpu pointer without changing the internal state */
    T *getRawGpuPtr() 
    {
      allocateGpuIfNecessary();
      return m_dvce_base;
    }


    const T *getROPtr() const
    {
      enableCpuRead();
      return m_host_base;
    }

    T *getPtr()
    {
      enableCpuWrite();
      return m_host_base;
    }

    const T *getROGpuPtr() const
    {
      enableGpuRead();
      return m_dvce_base;
    }

    T *getGpuPtr()
    {
      enableGpuWrite();
      return m_dvce_base;
    }

    // get write-exclusive CPU pointer
    T *getWEPtr() {
      allocateCpuIfNecessary();
      m_is_cpu_valid = true;
      m_is_gpu_valid = false;
      return m_host_base;
    }

    T *getWOGpuPtr() {
      allocateGpuIfNecessary();
      m_is_cpu_valid = false;
      m_is_gpu_valid = true;
      return m_dvce_base;
    }


    T &operator[](int index)
      {
        assert(index < m_size);
        T *host = getPtr();
        return host[index];
      }

    const T &operator[](int index) const
    {
      assert(index < m_size);
      const T *host = getROPtr();
      return host[index];
    }

    /* only dma what's needed, instead of the whole array */
    const T getElementAt(const int index) const
    {
      assert(index < m_size);
      assert(m_is_cpu_valid || m_is_gpu_valid);
      if (m_is_cpu_valid)
        return m_host_base[index];
      T ele; 
      allocateCpuIfNecessary();
      cudaError_t error = cudaMemcpy(&ele, m_dvce_base.get()+index, sizeof(T), cudaMemcpyDeviceToHost); 
      cassert(error == cudaSuccess);
      return ele;
    }

    void setElementAt(T &value, const int index)
    {
      cassert(index < m_size);
      cassert(m_is_cpu_valid || m_is_gpu_valid);
      if (m_is_cpu_valid)
        m_host_base[index] = value;
      if (m_is_gpu_valid)
        {
          cudaError_t error = cudaMemcpy(m_dvce_base.get()+index, &value, sizeof(T), cudaMemcpyHostToDevice); 
          cassert(error == cudaSuccess);
        }
    }

    void invalidateGpu() {
      m_is_gpu_valid = false;
    }

    void invalidateCpu() {
        m_is_cpu_valid = false;
    }


    inline typename thrust::device_ptr<T> gbegin()
    { 
      enableGpuWrite();
      return thrust::device_ptr<T>(getGpuPtr()); 
    }

    inline typename thrust::device_ptr<T> gend()
    { 
      enableGpuWrite();
      return thrust::device_ptr<T>(getGpuPtr()+m_size);
    }

    inline typename thrust::device_ptr<T> owbegin()
    { 
      return thrust::device_ptr<T>(getWOGpuPtr()); 
    }

    inline typename thrust::device_ptr<T> owend()
    { 
      return thrust::device_ptr<T>(getWOGpuPtr()+m_size);
    }


    inline typename thrust::device_ptr<T> gbegin() const
    { 
      enableGpuRead();
      return thrust::device_ptr<T>(const_cast<T*>(getROGpuPtr()));
    }

    inline typename thrust::device_ptr<T> gend() const
    {
      enableGpuRead();
      return thrust::device_ptr<T>(const_cast<T*>(getROGpuPtr()+m_size));
    }

    /*! explicitly sync to GPU buffer */
    void syncToGpu() const {
      //	assert(!(m_is_gpu_valid && !m_is_cpu_valid));
      allocateGpuIfNecessary();
      fromHostToDvce();
      m_is_gpu_valid = true;
    }

    /*! explicitly sync to CPU buffer */
    void syncToCpu() const {
      //	assert(!(m_is_cpu_valid && !m_is_gpu_valid));
      allocateCpuIfNecessary();
      fromDvceToHost();	
      m_is_cpu_valid = true;
    }

    inline void enableGpuRead() const
    {
      allocateGpuIfNecessary();
      if (!m_is_gpu_valid)
        {
          fromHostToDvceIfNecessary();
          setGpuAvailable();
        }
    }

    inline void enableGpuWrite() const
    {
      allocateGpuIfNecessary();
      if (!m_is_gpu_valid)
      fromHostToDvceIfNecessary();
      m_is_cpu_valid = false;
      m_is_gpu_valid = true;
    }

    inline void enableCpuRead() const
    {
      allocateCpuIfNecessary();
      if (!m_is_cpu_valid)
        {
          fromDvceToHostIfNecessary();
          m_is_cpu_valid = true;
        }
    }

    inline void enableCpuWrite() const
    {
      allocateCpuIfNecessary();
      if (!m_is_cpu_valid)
        fromDvceToHostIfNecessary();

      m_is_cpu_valid = true;
      m_is_gpu_valid = false;
    }

    void setGpuAvailable() const {
      m_is_gpu_valid = true;
    }


    private:
    void destroy() {
      if (m_dvce_base && gpuNeedToFree) {
        gHeapManager.Free(GPU_HEAP, m_dvce_base.get());
        m_dvce_base = NULL;
      }
      if (m_host_base && m_capacity > 0 && cpuNeedToFree) {
        gHeapManager.Free(CPU_HEAP, m_host_base.get());
        m_host_base = NULL;	  
      }
      init_state();
    }

    // deep copy from
    void deep_copy(const CktArray<T> &src) 
    {
      clean_resize(src.size());
      if (src.m_is_gpu_valid)
        {
          if (src.size())
            cudaMemcpy(getWOGpuPtr(), src.getROGpuPtr(), sizeof(T)*m_size, cudaMemcpyDefault);
        }
      else if (src.m_is_cpu_valid)
        {
          if (src.size())
            memcpy(getPtr(), src.getROPtr(), sizeof(T)*m_size);
        }
    }
    
    void init_state() {
      m_size = 0;
      m_capacity = 0;
      m_host_base.reset();
      m_dvce_base.reset();
      m_is_cpu_valid = false;
      m_is_gpu_valid = false;
      m_gpu_allocated = false;
      m_cpu_allocated = false;
    }
    
    inline void allocateCpuIfNecessary()  const
    {
      if (!m_cpu_allocated && m_size)
        {
          std::shared_ptr<T> newm_host_base((T*)GpuHostAllocator(m_size*sizeof(T)), GpuHostDeleter);
          m_host_base = newm_host_base;
          m_cpu_allocated = true;
        }
    }

    inline void allocateGpuIfNecessary() const
    {
      if (!m_gpu_allocated && m_size)
        {
          std::shared_ptr<T> newm_dvce_base((T*)GpuDeviceAllocator(m_size*sizeof(T)), GpuDeviceDeleter);
          assert(newm_dvce_base != 0);
          m_dvce_base = newm_dvce_base;
          m_gpu_allocated = true;
        }
    }


    inline void fromHostToDvce() const {
      if (m_size) {
        cassert(m_host_base);
        cassert(m_dvce_base);
        cudaError_t error = cudaMemcpy(m_dvce_base, m_host_base, m_size* sizeof(T), cudaMemcpyHostToDevice);
        cassert(error == cudaSuccess);
      }

    }
    
    inline void fromHostToDvceIfNecessary() const
    {
      if (m_is_cpu_valid && !m_is_gpu_valid)
      {
        fromHostToDvce();
      }
    }

    inline void fromDvceToHost() const
    {
      if (m_size) {
        cudaError_t error = cudaMemcpy(m_host_base, m_dvce_base, m_size * sizeof(T),cudaMemcpyDeviceToHost);
        cassert(error == cudaSuccess);
      }
    }
    
    inline void fromDvceToHostIfNecessary() const
    {
      if (m_is_gpu_valid && !m_is_cpu_valid)
        {
          if (size()) {
            cassert(m_dvce_base);
            cassert(m_host_base);
          }
          fromDvceToHost();
        }
    }

    int64_t m_size;
    int m_capacity;
    std::shared_ptr<T> m_host_base;
    std::shared_ptr<T> m_dvce_base;
    mutable bool m_is_cpu_valid = false;
    mutable bool m_is_gpu_valid = false;
    mutable bool m_gpu_allocated = false;
    mutable bool m_cpu_allocated = false;
    mutable bool gpuNeedToFree = true;
    mutable bool cpuNeedToFree = true;      

  };


  template <typename T, int BATCH>
  struct BatchInit {
    T *ptrs[BATCH];
    size_t sizes[BATCH];
    T vals[BATCH];

    T *big_ptrs[BATCH];
    size_t big_sizes[BATCH];
    T vals2[BATCH];
  };

  template <typename T, int BATCH>
  void batch_fill_wrapper(int num_small_arrays, int num_big_arrays, const BatchInit<T, BATCH> &init, cudaStream_t stream);

  /*! Help class to initialize multiple vectors at the same time
    *  
    */
  template <class T, int BATCH>
  class BatchInitializer {
  public:
    
    void push_back(CktArray<T> *array, T val) {
      m_arrays.push_back(array);
      m_vals.push_back(val);
      assert(m_arrays.size() < BATCH);
    }
    void init(cudaStream_t stream = 0) {
      BatchInit<T, BATCH> init;
      memset(&init, 0, sizeof(init));
      if (m_arrays.size() > BATCH)
        std::cerr << "Number of arrays " << m_arrays.size() << 
          " exceeding template BATCH " << BATCH << ", please increase BATCH." << std::endl;
      int small_idx = 0, big_idx = 0;
      for (int i = 0; i != m_arrays.size(); i++) {
        size_t _size = m_arrays[i]->size();
        if (_size < 100000)  {
          init.ptrs[small_idx] = m_arrays[i]->getWOGpuPtr();
          init.sizes[small_idx] = _size;
          init.vals[small_idx] = m_vals[i];
          ++small_idx;
        } else {
          init.big_ptrs[big_idx] = m_arrays[i]->getWOGpuPtr();
          init.big_sizes[big_idx] = _size;
          init.vals2[big_idx] = m_vals[i];
          ++big_idx;
        }
      }
      batch_fill_wrapper<T, BATCH>(small_idx, big_idx, init, stream);
    }


  private:
    std::vector<CktArray<T> *> m_arrays;
    std::vector<T> m_vals;
  };

  // aliasing C++11 feature
  template <typename T> 
  using CVector = CktArray<T>;
  //#define CVector ckt::CktArray 
} // ckt
