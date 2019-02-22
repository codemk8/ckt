#include <cstdlib>
#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include "heap_manager.hpp"
#include "heap_allocator.hpp"
#include "utility.hpp"
using namespace std;

//#define USE_CUDA_ALLOCATOR
#undef USE_CUDA_ALLOCATOR
namespace ckt {
  HeapManager gHeapManager;
  int HeapManager::max_device_ids = 32;

  HeapManager::HeapManager() {
    m_gpu_heap_allocator.resize(HeapManager::max_device_ids, nullptr);
  }

  HeapManager::~HeapManager()
  {
    for (auto i = m_gpu_heap_allocator.begin(); i != m_gpu_heap_allocator.end(); i++) {
      if (*i != nullptr)
        delete *i;
    }
  }

  void HeapManager::Malloc(Memory_Type type, void **addr, const size_t &size)
  {
    if (type == CPU_HEAP)
      {
        *addr = (void *)malloc(size);
#ifdef _DEBUG
        mCpuMemoryTracker.insert( pair<void *, int>(*addr, size));
        curCpuUsage += size;
        maxCpuUsage = maxCpuUsage > curCpuUsage? maxCpuUsage: curCpuUsage;
#endif      
      }
    else if (type == GPU_HEAP)
      {
        // init gpu allocator if not exist
#ifdef USE_CUDA_ALLOCATOR
        cudaMalloc(addr, size);
        if (size && (*addr == 0))  {
          check_cuda_error_always("cudaMemGetInfo", __FILE__, __LINE__);	
          int gpu = -1;
          cudaGetDevice(&gpu);
          size_t free, total;
          cudaMemGetInfo(&free, &total);
          check_cuda_error("cudaMemGetInfo", __FILE__, __LINE__);	
          printf("allocating memory size %ld failed on GPU %d, total %ld, free %ld\n", size, gpu, total, free);
        }
        check_cuda_error("cudaMalloc", __FILE__, __LINE__);
#else
        HeapAllocator *allocator = get_gpu_allocator();
        assert(allocator);
        *addr = allocator->allocate(size);
#endif
        if (addr == 0) {
          size_t free, total;
          cudaMemGetInfo(&free, &total);
          fprintf(stderr, "Failed to allocate memory size %f Kbytes, free memory %f Kbytes, total %f Kbytes.\n",
                  float(size)/1000., float(free)/1000., float(total)/1000.);
        }
      }
    else
      assert(0);
  }

  HeapAllocator *HeapManager::get_gpu_allocator()
  {
    int device(-1);
    cudaGetDevice(&device);
    assert(device >= 0);
    assert(device <= HeapManager::max_device_ids);
    if (m_gpu_heap_allocator[device] == nullptr)
      m_gpu_heap_allocator[device] = new HeapAllocator();
    return m_gpu_heap_allocator[device];
  }

  void HeapManager::Free(Memory_Type type, void *addr)
  {
    if (type == CPU_HEAP)
      {
        free(addr);

      }
    else if (type == GPU_HEAP)
      {
#ifdef USE_CUDA_ALLOCATOR
        cudaFree(addr);
#else
        get_gpu_allocator()->deallocate(addr);
#endif
      }
  }

  void *GpuHostAllocator(size_t size)
  {
    void *hostBase(0);

    gHeapManager.Malloc(CPU_HEAP, (void**)&hostBase, size);
    return hostBase;
  }

  void *GpuDeviceAllocator(size_t size)
  {
    void *dvceBase(0);
    gHeapManager.Malloc(GPU_HEAP, (void**)&dvceBase, size);
    return dvceBase;
  }

  void GpuHostDeleter(void *ptr)
  {
    gHeapManager.Free(CPU_HEAP, ptr);
  }

  void EmptyDeviceDeleter(void *ptr, size_t size)
  {
  }

  void GpuDeviceDeleter(void *ptr, size_t size)
  {
    gHeapManager.Free(GPU_HEAP, ptr);
  }

}
