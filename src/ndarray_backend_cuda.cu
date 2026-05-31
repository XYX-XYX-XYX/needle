#include <cstdint>
#include <cuda_runtime.h>
#include "flash_attention_common.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>


namespace needle {
namespace cuda {

#ifdef NEEDLE_ENABLE_FLASHATTN_STUB

enum class FlashAttentionKernelTarget { kAuto, kSm80, kSm90 };

FlashAttentionKernelTarget ParseFlashAttentionKernel(const std::string& kernel) {
  if (kernel == "auto") return FlashAttentionKernelTarget::kAuto;
  if (kernel == "sm80") return FlashAttentionKernelTarget::kSm80;
  if (kernel == "sm90") return FlashAttentionKernelTarget::kSm90;
  std::ostringstream msg;
  msg << "unknown flash attention kernel " << kernel << "; expected auto, sm80, or sm90";
  throw std::runtime_error(msg.str());
}

std::pair<int, int> GetCurrentCudaComputeCapability() {
  int device = 0;
  CudaCheck(cudaGetDevice(&device), "cudaGetDevice for flash attention kernel dispatch");

  cudaDeviceProp prop;
  CudaCheck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties for flash attention kernel dispatch");
  return {prop.major, prop.minor};
}

std::pair<int, int> GetCurrentCudaComputeCapabilityNoThrow() {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return {-1, -1};
  }

  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return {-1, -1};
  }
  return {prop.major, prop.minor};
}

FlashAttentionKernelTarget ResolveFlashAttentionKernel(const std::string& kernel) {
  FlashAttentionKernelTarget target = ParseFlashAttentionKernel(kernel);
  if (target != FlashAttentionKernelTarget::kAuto) {
    return target;
  }

  auto capability = GetCurrentCudaComputeCapability();
  return capability.first >= 9 ? FlashAttentionKernelTarget::kSm90 : FlashAttentionKernelTarget::kSm80;
}

const char* FlashAttentionKernelTargetName(FlashAttentionKernelTarget target) {
  return target == FlashAttentionKernelTarget::kSm90 ? "sm90" : "sm80";
}

std::string ResolveFlashAttentionKernelName(const std::string& kernel) {
  return FlashAttentionKernelTargetName(ResolveFlashAttentionKernel(kernel));
}

void LaunchFlashAttentionForward(const CudaArray& q, const CudaArray& k, const CudaArray& v,
                                 CudaArray* out, size_t batch_size, size_t num_heads,
                                 size_t q_len, size_t kv_len,
                                 size_t head_dim, float dropout, bool causal,
                                 const std::string& kernel, cudaStream_t stream) {
  FlashAttentionKernelTarget target = ResolveFlashAttentionKernel(kernel);
  if (target == FlashAttentionKernelTarget::kSm90) {
    LaunchFlashAttentionForwardSm90(q, k, v, out, batch_size, num_heads, q_len, kv_len,
                                    head_dim, dropout, causal, stream);
    return;
  }

  LaunchFlashAttentionForwardSm80(q, k, v, out, batch_size, num_heads, q_len, kv_len,
                                  head_dim, dropout, causal, stream);
}

void FlashAttentionForward(const CudaArray& q, const CudaArray& k, const CudaArray& v,
                          CudaArray* out, size_t batch_size, size_t num_heads,
                          size_t q_len, size_t kv_len,
                          size_t head_dim, float dropout, bool causal,
                          const std::string& kernel) {
  LaunchFlashAttentionForward(q, k, v, out, batch_size, num_heads, q_len, kv_len,
                              head_dim, dropout, causal, kernel, 0);
}

float FlashAttentionForwardBenchmark(const CudaArray& q, const CudaArray& k, const CudaArray& v,
                                     CudaArray* out, size_t batch_size, size_t num_heads,
                                     size_t q_len, size_t kv_len,
                                     size_t head_dim, float dropout, bool causal,
                                     const std::string& kernel, int repeats) {
  if (repeats <= 0) {
    throw std::runtime_error("flash attention benchmark repeats must be positive");
  }

  cudaStream_t stream = 0;
  cudaEvent_t start;
  cudaEvent_t stop;
  CudaCheck(cudaEventCreate(&start), "cudaEventCreate(start)");
  CudaCheck(cudaEventCreate(&stop), "cudaEventCreate(stop)");

  try {
    CudaCheck(cudaEventRecord(start, stream), "cudaEventRecord(start)");
    for (int i = 0; i < repeats; ++i) {
      LaunchFlashAttentionForward(q, k, v, out, batch_size, num_heads, q_len, kv_len,
                                  head_dim, dropout, causal, kernel, stream);
    }
    CudaCheck(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
    CudaCheck(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    CudaCheck(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    CudaCheck(cudaEventDestroy(start), "cudaEventDestroy(start)");
    CudaCheck(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    return elapsed_ms;
  } catch (...) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    throw;
  }
}
#endif

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace needle::cuda;;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    uint16_t* host_ptr = static_cast<uint16_t*>(std::malloc(a.size * ELEM_SIZE));
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // Return a NumPy float16 array. pybind11 does not have built-in NumPy
    // type info for cutlass::half_t, but the underlying binary layout matches.
    py::dtype float16_dtype = py::module_::import("numpy").attr("dtype")("float16");
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array(float16_dtype, shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array a, CudaArray* out) {
    py::dtype float16_dtype = py::module_::import("numpy").attr("dtype")("float16");
    py::array host_array =
        py::module_::import("numpy").attr("ascontiguousarray")(a, float16_dtype);
    py::buffer_info buf = host_array.request();
    if (static_cast<size_t>(buf.size) != out->size) {
      throw std::runtime_error("from_numpy size mismatch for CUDA float16 array");
    }
    cudaError_t err =
        cudaMemcpy(out->ptr, buf.ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  // m.def("fill", Fill);
  // m.def("compact", Compact);
  // m.def("ewise_setitem", EwiseSetitem);
  // m.def("scalar_setitem", ScalarSetitem);
  // m.def("ewise_add", EwiseAdd);
  // m.def("scalar_add", ScalarAdd);

  // m.def("ewise_mul", EwiseMul);
  // m.def("scalar_mul", ScalarMul);
  // m.def("ewise_div", EwiseDiv);
  // m.def("scalar_div", ScalarDiv);
  // m.def("scalar_power", ScalarPower);

  // m.def("ewise_maximum", EwiseMaximum);
  // m.def("scalar_maximum", ScalarMaximum);
  // m.def("ewise_eq", EwiseEq);
  // m.def("scalar_eq", ScalarEq);
  // m.def("ewise_ge", EwiseGe);
  // m.def("scalar_ge", ScalarGe);

  // m.def("ewise_log", EwiseLog);
  // m.def("ewise_exp", EwiseExp);
  // m.def("ewise_tanh", EwiseTanh);

  // m.def("matmul", Matmul);

  // m.def("reduce_max", ReduceMax);
  // m.def("reduce_sum", ReduceSum);

#ifdef NEEDLE_ENABLE_FLASHATTN_STUB
  m.attr("__has_flash_attention_stub__") = true;
  m.attr("__flash_attention_kernels__") = std::vector<std::string>{"auto", "sm80", "sm90"};
  auto capability = GetCurrentCudaComputeCapabilityNoThrow();
  m.attr("__cuda_compute_capability__") = py::make_tuple(capability.first, capability.second);
  m.def("flash_attention_resolve_kernel", ResolveFlashAttentionKernelName);
  m.def("flash_attention_forward", FlashAttentionForward);
  m.def("flash_attention_forward_benchmark", FlashAttentionForwardBenchmark);
#else
  m.attr("__has_flash_attention_stub__") = false;
  m.attr("__flash_attention_kernels__") = std::vector<std::string>{};
  m.attr("__cuda_compute_capability__") = py::make_tuple(-1, -1);
#endif
}
