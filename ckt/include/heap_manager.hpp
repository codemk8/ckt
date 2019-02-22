#pragma once

#include <map>
#include <vector>

//#define _DEBUG
namespace ckt {
/* a heap manager that helps debugging memory alloc/free bugs */
typedef enum{
  CPU_HEAP,
  GPU_HEAP
} Memory_Type;

void *GpuHostAllocator(size_t size);
void *GpuDeviceAllocator(size_t size);
  void GpuHostDeleter(void *ptr, size_t size);
  void GpuDeviceDeleter(void *ptr, size_t size);
  void EmptyDeviceDeleter(void *ptr, size_t size);

  class HeapAllocator;
class HeapManager{
 public:
  HeapManager();
  ~HeapManager();

  void Malloc(Memory_Type type, void ** addr, const size_t &size);
  void Free(Memory_Type type, void *addr);
  int find(Memory_Type type, void *addr);
 private:
  HeapAllocator *get_gpu_allocator();
  static int max_device_ids;
  // for multiple GPUs
  std::vector<HeapAllocator *> m_gpu_heap_allocator;
  std::vector<size_t> m_allocatedSize;
#ifdef _DEBUG
  int count = 0;
  int maxCpuUsage = 0;
  int maxGpuUsage = 0;
  int curCpuUsage = 0;
  int curGpuUsage = 0;
#endif
};

}
