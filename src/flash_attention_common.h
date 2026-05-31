#pragma once

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256
#define TILE 4

constexpr size_t ELEM_SIZE = sizeof(uint16_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }

  ~CudaArray() { cudaFree(ptr); }

  size_t ptr_as_int() { return reinterpret_cast<size_t>(ptr); }

  void* ptr;
  size_t size;
};

inline void CudaCheck(cudaError_t err, const char* context) {
  if (err != cudaSuccess) {
    std::ostringstream msg;
    msg << context << ": " << cudaGetErrorString(err);
    throw std::runtime_error(msg.str());
  }
}

void LaunchFlashAttentionForwardSm80(const CudaArray& q, const CudaArray& k, const CudaArray& v,
                                     CudaArray* out, size_t batch_size, size_t num_heads,
                                     size_t q_len, size_t kv_len,
                                     size_t head_dim, float dropout, bool causal,
                                     cudaStream_t stream);

void LaunchFlashAttentionForwardSm90(const CudaArray& q, const CudaArray& k, const CudaArray& v,
                                     CudaArray* out, size_t batch_size, size_t num_heads,
                                     size_t q_len, size_t kv_len,
                                     size_t head_dim, float dropout, bool causal,
                                     cudaStream_t stream);

}  // namespace cuda
}  // namespace needle
