#pragma once
#include <cuda_runtime_api.h>

namespace ckt 
{
  namespace cuda 
  {
    /*! \brief defines are used in device code
     */
    #define CKT_cuda_warpsize_shift  5
    #define CKT_cuda_warpsize_mask   0x1F
    #define CKT_cuda_blocksize       512
    #define CKT_cuda_blocksize_shift 9
    #define CKT_cuda_bs_mask         0xFF
    #define CKT_cuda_bs_mask2        0xFFFFFF00
    #define CKT_cuda_max_blocks      120
    #define CKT_cuda_warpsize        32

    class CKTKonst {
    public:
      static const int cuda_blocksize;
      static const int cuda_max_blocks;
      static const int cuda_warpsize;
      static const int cuda_warpsize_shift;
      static const int cuda_warpsize_mask;

    };

    __inline__ void get_cuda_property(cudaDeviceProp &property) {
      int gpu; cudaGetDevice(&gpu);
      cudaGetDeviceProperties(&property, gpu);
    }

  }
}


#define CAP_BLOCK_SIZE(block) (block > ckt::cuda::CKTKonst::cuda_max_blocks ? ckt::cuda::CKTKonst::cuda_max_blocks:block)
#define GET_BLOCKS(N, bs) CAP_BLOCK_SIZE( (N + bs -1 )/bs)
#define JCUDA_BS (ckt::cuda::CKTKonst::cuda_blocksize)

// used in kernels 
#define kernel_get_1d_gid  (blockIdx.x*blockDim.x + threadIdx.x)
#define kernel_get_1d_stride (blockDim.x * gridDim.x)


 


