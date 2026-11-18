#ifndef CUZFP_DECODE2_CUH
#define CUZFP_DECODE2_CUH

#include "shared.h"
#include "decode.cuh"
#include "type_info.cuh"

namespace cuZFP {

template<typename Scalar> 
__device__ __host__ inline 
void scatter_partial2(const Scalar* q, Scalar* p, int nx, int ny, int sx, int sy)
{
  uint x, y;
  for (y = 0; y < 4; y++)
    if (y < ny) {
      for (x = 0; x < 4; x++)
        if (x < nx) {
          *p = q[4 * y + x];
          p += sx;
        }
      p += sy - nx * sx;
    }
}

template<typename Scalar> 
__device__ __host__ inline 
void scatter2(const Scalar* q, Scalar* p, int sx, int sy)
{
  uint x, y;
  for (y = 0; y < 4; y++, p += sy - 4 * sx)
    for (x = 0; x < 4; x++, p += sx)
      *p = *q++;
}


template<class Scalar, int BlockSize>
__global__
void
cudaDecode2(Word *blocks,
            Scalar *out,
            const uint2 dims,
            const int2 stride,
            const uint2 padded_dims,
            uint maxbits)
{
  typedef unsigned long long int ull;
  typedef long long int ll;
  const ull blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x * gridDim.y * blockIdx.z;

  // each thread gets a block so the block index is 
  // the global thread index
  const ull block_idx = blockId * blockDim.x + threadIdx.x;
  
  const int total_blocks = (padded_dims.x * padded_dims.y) / 16; 
  
  if(block_idx >= total_blocks) 
  {
    return;
  }

  BlockReader<BlockSize> reader(blocks, maxbits, block_idx, total_blocks);
 
  Scalar result[BlockSize];
  memset(result, 0, sizeof(Scalar) * BlockSize);

  zfp_decode(reader, result, maxbits);

  // logical block dims
  uint2 block_dims;
  block_dims.x = padded_dims.x >> 2; 
  block_dims.y = padded_dims.y >> 2; 
  // logical pos in 3d array
  uint2 block;
  block.x = (block_idx % block_dims.x) * 4; 
  block.y = ((block_idx/ block_dims.x) % block_dims.y) * 4; 
  
  const ll offset = (ll)block.x * stride.x + (ll)block.y * stride.y; 

  bool partial = false;
  if(block.x + 4 > dims.x) partial = true;
  if(block.y + 4 > dims.y) partial = true;
  if(partial)
  {
    const uint nx = block.x + 4 > dims.x ? dims.x - block.x : 4;
    const uint ny = block.y + 4 > dims.y ? dims.y - block.y : 4;
    scatter_partial2(result, out + offset, nx, ny, stride.x, stride.y);
  }
  else
  {
    scatter2(result, out + offset, stride.x, stride.y);
  }
}

// fixed-precision (variable-rate) counterpart of cudaDecode2; see
// cudaDecode3Precision in decode3.cuh for the rationale.
template<class Scalar, int BlockSize>
__global__
void
cudaDecode2Precision(Word *blocks,
                      const uint64 *block_offsets,
                      Scalar *out,
                      const uint2 dims,
                      const int2 stride,
                      const uint2 padded_dims,
                      int maxprec,
                      int minexp,
                      uint num_blocks)
{
  typedef unsigned long long int ull;
  typedef long long int ll;
  const ull blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x * gridDim.y * blockIdx.z;

  const ull block_idx = blockId * blockDim.x + threadIdx.x;

  if(block_idx >= num_blocks)
  {
    return;
  }

  BlockReader<BlockSize> reader(blocks, block_offsets[block_idx], block_idx, num_blocks);

  Scalar result[BlockSize];
  memset(result, 0, sizeof(Scalar) * BlockSize);

  zfp_decode_precision<Scalar, BlockSize>(reader, result, maxprec, minexp, 2);

  // logical block dims
  uint2 block_dims;
  block_dims.x = padded_dims.x >> 2;
  block_dims.y = padded_dims.y >> 2;
  // logical pos in 2d array
  uint2 block;
  block.x = (block_idx % block_dims.x) * 4;
  block.y = ((block_idx/ block_dims.x) % block_dims.y) * 4;

  const ll offset = (ll)block.x * stride.x + (ll)block.y * stride.y;

  bool partial = false;
  if(block.x + 4 > dims.x) partial = true;
  if(block.y + 4 > dims.y) partial = true;
  if(partial)
  {
    const uint nx = block.x + 4 > dims.x ? dims.x - block.x : 4;
    const uint ny = block.y + 4 > dims.y ? dims.y - block.y : 4;
    scatter_partial2(result, out + offset, nx, ny, stride.x, stride.y);
  }
  else
  {
    scatter2(result, out + offset, stride.x, stride.y);
  }
}

template<class Scalar>
void decode2launch_precision(uint2 dims,
                              int2 stride,
                              Word *stream,
                              const uint64 *d_block_offsets,
                              size_t num_blocks,
                              Scalar *d_data,
                              int maxprec,
                              int minexp)
{
  const int cuda_block_size = 128;
  dim3 block_size = dim3(cuda_block_size, 1, 1);

  uint2 zfp_pad(dims);
  if(zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if(zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;

  int block_pad = 0;
  if(num_blocks % cuda_block_size != 0)
  {
    block_pad = cuda_block_size - num_blocks % cuda_block_size;
  }

  size_t launch_blocks = block_pad + num_blocks;
  dim3 grid_size = calculate_grid_size(launch_blocks, cuda_block_size);

  cudaDecode2Precision<Scalar, 16> << < grid_size, block_size >> >
    (stream,
     d_block_offsets,
     d_data,
     dims,
     stride,
     zfp_pad,
     maxprec,
     minexp,
     (uint)num_blocks);
}

template<class Scalar>
void decode2_precision(uint2 dims,
                        int2 stride,
                        Word *stream,
                        const uint64 *d_block_offsets,
                        size_t num_blocks,
                        Scalar *d_data,
                        int maxprec,
                        int minexp)
{
  decode2launch_precision<Scalar>(dims, stride, stream, d_block_offsets, num_blocks, d_data, maxprec, minexp);
}

template<class Scalar>
size_t decode2launch(uint2 dims,
                     int2 stride,
                     Word *stream,
                     Scalar *d_data,
                     uint maxbits)
{
  const int cuda_block_size = 128;
  dim3 block_size;
  block_size = dim3(cuda_block_size, 1, 1);
  
  uint2 zfp_pad(dims); 
  // ensure that we have block sizes
  // that are a multiple of 4
  if(zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if(zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;

  const int zfp_blocks = (zfp_pad.x * zfp_pad.y) / 16; 

  
  //
  // we need to ensure that we launch a multiple of the 
  // cuda block size
  //
  int block_pad = 0; 
  if(zfp_blocks % cuda_block_size != 0)
  {
    block_pad = cuda_block_size - zfp_blocks % cuda_block_size; 
  }


  size_t stream_bytes = calc_device_mem2d(zfp_pad, maxbits);
  size_t total_blocks = block_pad + zfp_blocks;
  dim3 grid_size = calculate_grid_size(total_blocks, cuda_block_size);

#ifdef CUDA_ZFP_RATE_PRINT
  // setup some timing code
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
#endif

  cudaDecode2<Scalar, 16> << < grid_size, block_size >> >
    (stream,
		 d_data,
     dims,
     stride,
     zfp_pad,
     maxbits);

#ifdef CUDA_ZFP_RATE_PRINT
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
	cudaStreamSynchronize(0);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float seconds = milliseconds / 1000.f;
  float rate = (float(dims.x * dims.y) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("Decode elapsed time: %.5f (s)\n", seconds);
  printf("# decode2 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif
  return stream_bytes;
}

template<class Scalar>
size_t decode2(uint2 dims, 
               int2 stride,
               Word *stream,
               Scalar *d_data,
               uint maxbits)
{
	return decode2launch<Scalar>(dims, stride, stream, d_data, maxbits);
}

} // namespace cuZFP

#endif
