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
        mSize(size),
        mCapacity(0),
        hostBase(),
        dvceBase(),
        isCpuValid(false),
        isGpuValid(false),
        gpuAllocated(false),
        cpuAllocated(false),
        isGpuArray(true),
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

    void SetGpuArray() {
      isGpuArray = true;
    }

    void SetCpuArray() {
      isGpuArray = false;
    }

    bool IsGpuArray() const {
      return isGpuArray;
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
      return isGpuValid;
    }

    bool CpuHasLatest() const {
      return isCpuValid;
    }
    
    // deep copy to
    void clone(CktArray<T> &dst) const
    {
      dst.clean_resize(size());
      if (isGpuValid)
        {
          cudaMemcpy(dst.getGpuPtr(), getROGpuPtr(), sizeof(T)*mSize, cudaMemcpyDeviceToDevice);
        }
      if (isCpuValid)
        {
          memcpy(dst.getPtr(), getROPtr(), sizeof(T)*mSize);
        }
      
      dst.isCpuValid = isCpuValid;
      dst.isGpuValid = isGpuValid;
      dst.gpuAllocated = gpuAllocated;
      dst.cpuAllocated = cpuAllocated;
    }

    void alias(const CktArray<T> & dst) {
      shallow_copy(dst);
      mCapacity = -1; // to disable calling free
    }
    
    void clear()
    {
      Resize(0);
    }

    int size() const
    {
      return mSize;
    }
    
    /*! A clean version of resize.
      It does not copy the old data, 
      nor does it initialize the data 
    */
    void clean_resize(int64_t _size) {
      /* TODO Reallocate if too waste of space */
      if (mCapacity >= _size || _size == 0)  {
        mSize = _size;
        isGpuValid = false;
        isCpuValid = false;
        return;
      }
      if (_size > 0)  {
        // depending on previous state
        if (gpuAllocated) {
          if (dvceBase)
            gHeapManager.Free(GPU_HEAP, dvceBase);
          gHeapManager.Malloc(GPU_HEAP, (void**)&dvceBase, _size * sizeof(T));
          isGpuValid = false;
        }
        if (cpuAllocated) {
          if (hostBase)
            gHeapManager.Free(CPU_HEAP, hostBase);
          gHeapManager.Malloc(CPU_HEAP, (void**)&hostBase, _size * sizeof(T));
          isCpuValid = false;
        }
        mSize = _size;
        mCapacity = _size;
      }
    }

    void Resize(int64_t new_size) 
    {
      if (new_size <= mCapacity)
      {
        // free memory if resize to zero
        if (new_size == 0 && mSize > 0) {
          destroy();
        }
        mSize = new_size;
      }
      else // need to reallocate
      {    
        if (!gpuAllocated && !cpuAllocated) {
          mSize = new_size;
        }
        else  {
          if (gpuAllocated)
          {
              std::shared_ptr<T> newDvceBase((T*)GpuDeviceAllocator(new_size*sizeof(T)), GpuDeviceDeleter);
              if (isGpuValid)
                {
                  cudaError_t error = cudaMemcpy(newDvceBase, dvceBase, mSize*sizeof(T), cudaMemcpyDeviceToDevice);
                  assert(error == cudaSuccess);
                }
              dvceBase = newDvceBase;
          }
          if (cpuAllocated)
          {
              std::shared_ptr<T> newHostBase((T*)GpuHostAllocator(new_size*sizeof(T)), GpuHostDeleter);
              if (isCpuValid)
                memcpy(newHostBase, hostBase, mSize*sizeof(T));
              hostBase = newHostBase;            
          }
        }
        mSize = new_size;
        mCapacity = new_size;
      }
    }

    /*! return the pointer without changing the internal state */
    T *getRawPtr() 
    {
      allocateCpuIfNecessary();
      return hostBase;
    }

    /*! return the gpu pointer without changing the internal state */
    T *getRawGpuPtr() 
    {
      allocateGpuIfNecessary();
      return dvceBase;
    }


    const T *getROPtr() const
    {
      enableCpuRead();
      return hostBase;
    }

    T *getPtr()
    {
      enableCpuWrite();
      return hostBase;
    }

    const T *getROGpuPtr() const
    {
      enableGpuRead();
      return dvceBase;
    }

    T *getGpuPtr()
    {
      enableGpuWrite();
      return dvceBase;
    }

    // get write-exclusive CPU pointer
    T *getWEPtr() {
      allocateCpuIfNecessary();
      isCpuValid = true;
      isGpuValid = false;
      return hostBase;
    }

    T *getWOGpuPtr() {
      allocateGpuIfNecessary();
      isCpuValid = false;
      isGpuValid = true;
      return dvceBase;
    }


    T &operator[](int index)
      {
        assert(index < mSize);
        T *host = getPtr();
        return host[index];
      }

    const T &operator[](int index) const
    {
      assert(index < mSize);
      const T *host = getROPtr();
      return host[index];
    }

    /* only dma what's needed, instead of the whole array */
    const T getElementAt(const int index) const
    {
      assert(index < mSize);
      assert(isCpuValid || isGpuValid);
      if (isCpuValid)
        return hostBase[index];
      T ele; 
      allocateCpuIfNecessary();
      cudaError_t error = cudaMemcpy(&ele, dvceBase.get()+index, sizeof(T), cudaMemcpyDeviceToHost); 
      cassert(error == cudaSuccess);
      return ele;
    }

    void setElementAt(T &value, const int index)
    {
      cassert(index < mSize);
      cassert(isCpuValid || isGpuValid);
      if (isCpuValid)
        hostBase[index] = value;
      if (isGpuValid)
        {
          cudaError_t error = cudaMemcpy(dvceBase.get()+index, &value, sizeof(T), cudaMemcpyHostToDevice); 
          cassert(error == cudaSuccess);
        }
    }

    void invalidateGpu() {
      isGpuValid = false;
    }

    void invalidateCpu() {
        isCpuValid = false;
    }


    inline typename thrust::device_ptr<T> gbegin()
    //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer)); }
    { 
      enableGpuWrite();
      return thrust::device_ptr<T>(getGpuPtr()); 
    }

    /*! \brief Return the last iterator (the first invalid iterator) in the srt::vector */
    inline typename thrust::device_ptr<T> gend()
    //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer+m_size)); }
    { 
      enableGpuWrite();
      return thrust::device_ptr<T>(getGpuPtr()+mSize);
    }

    inline typename thrust::device_ptr<T> owbegin()
    //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer)); }
    { 
      return thrust::device_ptr<T>(getWOGpuPtr()); 
    }

    /*! \brief Return the last iterator (the first invalid iterator) in the srt::vector */
    inline typename thrust::device_ptr<T> owend()
    //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer+m_size)); }
    { 
      return thrust::device_ptr<T>(getWOGpuPtr()+mSize);
    }


    /*! \brief Return the iterator to the first element in the srt::vector */
    inline typename thrust::device_ptr<T> gbegin() const
    //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer)); }
    { 
      enableGpuRead();
      return thrust::device_ptr<T>(const_cast<T*>(getROGpuPtr()));
    }

    /*! \brief Return the last iterator (the first invalid iterator) in the srt::vector */
    inline typename thrust::device_ptr<T> gend() const
    //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer+m_size)); }
    {
      enableGpuRead();
      return thrust::device_ptr<T>(const_cast<T*>(getROGpuPtr()+mSize));
    }

    /*! explicitly sync to GPU buffer */
    void syncToGpu() const {
      //	assert(!(isGpuValid && !isCpuValid));
      allocateGpuIfNecessary();
      fromHostToDvce();
      isGpuValid = true;
    }

    /*! explicitly sync to CPU buffer */
    void syncToCpu() const {
      //	assert(!(isCpuValid && !isGpuValid));
      allocateCpuIfNecessary();
      fromDvceToHost();	
      isCpuValid = true;
    }

    inline void enableGpuRead() const
    {
      allocateGpuIfNecessary();
      if (!isGpuValid)
        {
          fromHostToDvceIfNecessary();
          setGpuAvailable();
        }
    }

    inline void enableGpuWrite() const
    {
      allocateGpuIfNecessary();
      if (!isGpuValid)
      fromHostToDvceIfNecessary();
      isCpuValid = false;
      isGpuValid = true;
    }

    inline void enableCpuRead() const
    {
      allocateCpuIfNecessary();
      if (!isCpuValid)
        {
          fromDvceToHostIfNecessary();
          isCpuValid = true;
        }
    }

    inline void enableCpuWrite() const
    {
      allocateCpuIfNecessary();
      if (!isCpuValid)
        fromDvceToHostIfNecessary();

      isCpuValid = true;
      isGpuValid = false;
    }

    void setGpuAvailable() const {
      isGpuValid = true;
    }


    private:
    void destroy() {
      if (dvceBase && gpuNeedToFree) {
        gHeapManager.Free(GPU_HEAP, dvceBase.get());
        dvceBase = NULL;
      }
      if (hostBase && mCapacity > 0 && cpuNeedToFree) {
        gHeapManager.Free(CPU_HEAP, hostBase.get());
        hostBase = NULL;	  
      }
      init_state();
    }

    // deep copy from
    void deep_copy(const CktArray<T> &src) 
    {
      clean_resize(src.size());
      if (src.isGpuValid)
        {
          if (src.size())
            cudaMemcpy(getWOGpuPtr(), src.getROGpuPtr(), sizeof(T)*mSize, cudaMemcpyDefault);
        }
      else if (src.isCpuValid)
        {
          if (src.size())
            memcpy(getPtr(), src.getROPtr(), sizeof(T)*mSize);
        }
      isGpuArray = src.isGpuArray;
    }
    
    void init_state() {
      mSize = 0;
      mCapacity = 0;
      hostBase.reset();
      dvceBase.reset();
      isCpuValid = false;
      isGpuValid = false;
      gpuAllocated = false;
      cpuAllocated = false;
      isGpuArray = false;
    }
    
    inline void allocateCpuIfNecessary()  const
    {
      if (!cpuAllocated && mSize)
        {
          std::shared_ptr<T> newHostBase((T*)GpuHostAllocator(mSize*sizeof(T)), GpuHostDeleter);
          hostBase = newHostBase;
          cpuAllocated = true;
        }
    }

    inline void allocateGpuIfNecessary() const
    {
      if (!gpuAllocated && mSize)
        {
          std::shared_ptr<T> newDvceBase((T*)GpuDeviceAllocator(mSize*sizeof(T)), GpuDeviceDeleter);
          assert(newDvceBase != 0);
          dvceBase = newDvceBase;
          gpuAllocated = true;
        }
    }


    inline void fromHostToDvce() const {
      if (mSize) {
        cassert(hostBase);
        cassert(dvceBase);
        cudaError_t error = cudaMemcpy(dvceBase, hostBase, mSize* sizeof(T), cudaMemcpyHostToDevice);
        cassert(error == cudaSuccess);
      }

    }
    
    inline void fromHostToDvceIfNecessary() const
    {
      if (isCpuValid && !isGpuValid)
      {
        fromHostToDvce();
      }
    }

    inline void fromDvceToHost() const
    {
      if (mSize) {
        cudaError_t error = cudaMemcpy(hostBase, dvceBase, mSize * sizeof(T),cudaMemcpyDeviceToHost);
        cassert(error == cudaSuccess);
      }
    }
    
    inline void fromDvceToHostIfNecessary() const
    {
      if (isGpuValid && !isCpuValid)
        {
          if (size()) {
            cassert(dvceBase);
            cassert(hostBase);
          }
          fromDvceToHost();
        }
    }

    int64_t mSize;
    int mCapacity;
    std::shared_ptr<T> hostBase;
    std::shared_ptr<T> dvceBase;
    mutable bool isCpuValid = false;
    mutable bool isGpuValid = false;
    mutable bool gpuAllocated = false;
    mutable bool cpuAllocated = false;
    mutable bool isGpuArray = false;
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
