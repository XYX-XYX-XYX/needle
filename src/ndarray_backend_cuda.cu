#include <cstdint>
#include <cfloat>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

#ifdef NEEDLE_ENABLE_FLASHATTN_STUB
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#endif

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef cute::half_t scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

// struct CudaDims {
//   dim3 block, grid;
// };

// CudaDims CudaOneDim(size_t size) {
//   /**
//    * Utility function to get cuda dimensions for 1D call
//    */
//   CudaDims dim;
//   size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
//   dim.block = dim3(BASE_THREAD_NUM, 1, 1);
//   dim.grid = dim3(num_blocks, 1, 1);
//   return dim;
// }

// #define MAX_VEC_SIZE 8
// struct CudaVec {
//   uint32_t size;
//   int32_t data[MAX_VEC_SIZE];
// };

// CudaVec VecToCuda(const std::vector<int32_t>& x) {
//   CudaVec shape;
//   if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
//   shape.size = x.size();
//   for (size_t i = 0; i < x.size(); i++) {
//     shape.data[i] = x[i];
//   }
//   return shape;
// }

// ////////////////////////////////////////////////////////////////////////////////
// // Fill call
// ////////////////////////////////////////////////////////////////////////////////

// __global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (gid < size) out[gid] = val;
// }

// void Fill(CudaArray* out, scalar_t val) {
//   CudaDims dim = CudaOneDim(out->size);
//   FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
// }

// ////////////////////////////////////////////////////////////////////////////////
// // Compact and setitem cals
// ////////////////////////////////////////////////////////////////////////////////

// // Untility function to convert contiguous index i to memory location from strides
// __device__ static CudaVec GetIndices(CudaVec shape, size_t gid) 
// {

//   CudaVec indices;
//   indices.size = shape.size;
//   int idx = indices.size - 1;
//   for(; idx >= 0; idx--) {
//     indices.data[idx] = gid % shape.data[idx];
//     gid = gid / shape.data[idx];
//   }
//   return indices;

// }


// __global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
//                               CudaVec strides, size_t offset) {
//   /**
//    * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
//    * non-compact input a, to the corresponding item (at location gid) in the compact array out.
//    * 
//    * Args:
//    *   a: CUDA pointer to a array
//    *   out: CUDA point to out array
//    *   size: size of out array
//    *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
//    *   strides: vector of strides of out array
//    *   offset: offset of out array
//    */
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if(gid >= size) return;
//   /// BEGIN SOLUTION
//   auto out_ptr = out + gid;
//   CudaVec indices = GetIndices(shape, gid);
//   int32_t stride = 0;
//   for(size_t i = 0; i < indices.size; i++) {
//     stride += indices.data[i] * strides.data[i];
//   }
//   *out_ptr = a[offset + stride];
//   /// END SOLUTION
// }

// void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
//              std::vector<int32_t> strides, size_t offset) {
//   /**
//    * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
//    * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
//    * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
//    * the functions after this, however, you'll need to define these kernels as you see fit to 
//    * execute the underlying function.
//    * 
//    * Args:
//    *   a: non-compact represntation of the array, given as input
//    *   out: compact version of the array to be written
//    *   shape: shapes of each dimension for a and out
//    *   strides: strides of the *a* array (not out, which has compact strides)
//    *   offset: offset of the *a* array (not out, which has zero offset, being compact)
//    */

//   // Nothing needs to be added here
//   CudaDims dim = CudaOneDim(out->size);
//   CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
//                                          VecToCuda(strides), offset);
// }


// __global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
//                               CudaVec strides, size_t offset) 
// {

//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if(gid >= size) return;
//   auto out_ptr = out + offset;
//   auto a_ptr = a + gid;
//   CudaVec indices = GetIndices(shape, gid);
//   int32_t stride = 0;
//   for(size_t i = 0; i < indices.size; i++) {
//     stride += indices.data[i] * strides.data[i];
//   }
//   out_ptr[stride] = *a_ptr;
// }

// void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
//                   std::vector<int32_t> strides, size_t offset) {
//   /**
//    * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
//    * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
//    * 
//    * Args:
//    *   a: _compact_ array whose items will be written to out
//    *   out: non-compact array whose items are to be written
//    *   shape: shapes of each dimension for a and out
//    *   strides: strides of the *out* array (not a, which has compact strides)
//    *   offset: offset of the *out* array (not a, which has zero offset, being compact)
//    */
//   /// BEGIN SOLUTION
//   CudaDims dim = CudaOneDim(a.size);
//   EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
//                                          VecToCuda(strides), offset);
//   /// END SOLUTION
// }

// __global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
//                               CudaVec strides, size_t offset) 
// {

//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if(gid >= size) return;
//   auto out_ptr = out + offset;
//   CudaVec indices = GetIndices(shape, gid);
//   int32_t stride = 0;
//   for(size_t i = 0; i < indices.size; i++) {
//     stride += indices.data[i] * strides.data[i];
//   }
//   out_ptr[stride] = val;
// }

// void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
//                    std::vector<int32_t> strides, size_t offset) {
//   /**
//    * Set items is a (non-compact) array
//    * 
//    * Args:
//    *   size: number of elements to write in out array (note that this will note be the same as
//    *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
//    *         product of items in shape, but covenient to just pass it here.
//    *   val: scalar value to write to
//    *   out: non-compact array whose items are to be written
//    *   shape: shapes of each dimension of out
//    *   strides: strides of the out array
//    *   offset: offset of the out array
//    */
//   /// BEGIN SOLUTION
//   CudaDims dim = CudaOneDim(size);
//   ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
//                                          VecToCuda(strides), offset);
//   /// END SOLUTION
// }

// ////////////////////////////////////////////////////////////////////////////////
// // Elementwise and scalar operations
// ////////////////////////////////////////////////////////////////////////////////


// __global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
//   // Calculate the global index of the thread.
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (gid < size) out[gid] = a[gid] + b[gid];
// }

// void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
//   /**
//    * Add together two CUDA arrays.
//    * Args:
//    *   a: Input array 'a' to be added
//    *   b: Input array 'b' to be added
//    *   out: Output array to store the result of 'a + b'
//    */
//   CudaDims dim = CudaOneDim(out->size);

//   // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
//   EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
// }

// __global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
//   // Calculate the global index of the thread.
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (gid < size) out[gid] = a[gid] + val;
// }

// void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
//   /**
//    * Add a scalar value to every element of a CUDA array.
//    * Args:
//    *   a: Input array 'a'
//    *   val: Scalar value to be added
//    *   out: Output array to store the result of 'a + val'
//    */
//   CudaDims dim = CudaOneDim(out->size);

//   // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
//   // and store the result in array 'out'.
//   ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
// }

// /**
//  * In the code the follows, use the above template to create analogous elementise
//  * and and scalar operators for the following functions.  See the numpy backend for
//  * examples of how they should work.
//  *   - EwiseMul, ScalarMul
//  *   - EwiseDiv, ScalarDiv
//  *   - ScalarPower
//  *   - EwiseMaximum, ScalarMaximum
//  *   - EwiseEq, ScalarEq
//  *   - EwiseGe, ScalarGe
//  *   - EwiseLog
//  *   - EwiseExp
//  *   - EwiseTanh
//  *
//  * If you implement all these naively, there will be a lot of repeated code, so
//  * you are welcome (but not required), to use macros or templates to define these
//  * functions (however you want to do so, as long as the functions match the proper)
//  * signatures above.
//  */
 
// template <typename op> 
// __global__ void EwiseKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size, op operation)
// {
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (gid < size) out[gid] = operation(a[gid], b[gid]);
// }
// struct MulOp { __device__ scalar_t operator()(scalar_t a, scalar_t b) const { return a * b; } };
// struct DivOp { __device__ scalar_t operator()(scalar_t a, scalar_t b) const { return a / b; } };
// struct MaxOp { __device__ scalar_t operator()(scalar_t a, scalar_t b) const { return max(a, b); } };
// struct EqOp { __device__ scalar_t operator()(scalar_t a, scalar_t b) const { return a == b; } };
// struct GeOp { __device__ scalar_t operator()(scalar_t a, scalar_t b) const { return a >= b; } };
// struct PowerOp { __device__ scalar_t operator()(scalar_t a, scalar_t b) const { return pow(a, b); } };
// struct LogOp { __device__ scalar_t operator()(scalar_t a) const { return log(a); } };
// struct ExpOp { __device__ scalar_t operator()(scalar_t a) const { return exp(a); } };
// struct TanhOp { __device__ scalar_t operator()(scalar_t a) const { return tanh(a); } };

// void EwiseMul(const CudaArray &a, const CudaArray &b, CudaArray *out)
// {
//   CudaDims dim = CudaOneDim(out->size);
//   EwiseKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, MulOp());
// }

// void EwiseDiv(const CudaArray &a, const CudaArray &b, CudaArray *out)
// {
//   CudaDims dim = CudaOneDim(out->size);
//   EwiseKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, DivOp());
// }

// void EwiseMaximum(const CudaArray &a, const CudaArray &b, CudaArray *out)
// {
//   CudaDims dim = CudaOneDim(out->size);
//   EwiseKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, MaxOp());
// }

// void EwiseEq(const CudaArray &a, const CudaArray &b, CudaArray *out)
// {
//   CudaDims dim = CudaOneDim(out->size);
//   EwiseKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, EqOp());
// } 

// void EwiseGe(const CudaArray &a, const CudaArray &b, CudaArray *out)
// {
//   CudaDims dim = CudaOneDim(out->size);
//   EwiseKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, GeOp());
// } 

// template <typename op>
// __global__ void ScalarKernel(const scalar_t *a, const scalar_t val, scalar_t *out, size_t size, op operation)
// {
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if(gid < size) out[gid] = operation(a[gid], val);
// }

// void ScalarMul(const CudaArray &a, const scalar_t val, CudaArray *out)
// {
//   CudaDims dim = CudaOneDim(out->size);
//   ScalarKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, MulOp());
// }

// void ScalarDiv(const CudaArray &a, const scalar_t val, CudaArray *out)
// {
//   CudaDims dim = CudaOneDim(out->size);
//   ScalarKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, DivOp());
// }

// void ScalarPower(const CudaArray &a, const scalar_t val, CudaArray *out)
// {
//   CudaDims dim = CudaOneDim(out->size);
//   ScalarKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, PowerOp());
// }

// void ScalarMaximum(const CudaArray &a, const scalar_t val, CudaArray *out)
// {
//   CudaDims dim = CudaOneDim(out->size);
//   ScalarKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, MaxOp());
// }

// void ScalarEq(const CudaArray &a, const scalar_t val, CudaArray *out)
// {
//   CudaDims dim = CudaOneDim(out->size);
//   ScalarKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, EqOp());
// }

// void ScalarGe(const CudaArray &a, const scalar_t val, CudaArray *out)
// {
//   CudaDims dim = CudaOneDim(out->size);
//   ScalarKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, GeOp());
// }

// template <typename op>
// __global__ void UnaryKernel(const scalar_t *a, scalar_t *out, size_t size, op operation)
// {
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if(gid < size) out[gid] = operation(a[gid]);
// }

// void EwiseLog(const CudaArray &a, CudaArray *out)
// {
//   CudaDims dim = CudaOneDim(out->size);
//   UnaryKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, LogOp());
// }

// void EwiseExp(const CudaArray &a, CudaArray *out)
// {
//   CudaDims dim = CudaOneDim(out->size);
//   UnaryKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, ExpOp());
// }

// void EwiseTanh(const CudaArray &a, CudaArray *out)
// {
//   CudaDims dim = CudaOneDim(out->size);
//   UnaryKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, TanhOp());
// }
// ////////////////////////////////////////////////////////////////////////////////
// // Elementwise and scalar operations
// ////////////////////////////////////////////////////////////////////////////////

// __global__ void NaiveMatmulKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P)
// {
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if(gid >= M * P) return;
//   uint32_t idx = gid % P;
//   uint32_t idy = gid / P;

//   scalar_t sum = 0;
//   for(size_t i = 0; i < N; i++) {
//     sum += a[idy * N + i] * b[i * P + idx];
//   }
//   out[idy * P + idx] = sum;
// }

// #define BLOCK_DIM 32

// __global__ void ShareMemoryMatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N, uint32_t P) 
// {
//   int tx = threadIdx.x;
//   int ty = threadIdx.y;
//   int bx = blockIdx.x;
//   int by = blockIdx.y;

//   int row = by * BLOCK_DIM + ty;
//   int col = bx * BLOCK_DIM + tx;

//   scalar_t sum = 0;

//   __shared__ scalar_t As[BLOCK_DIM][BLOCK_DIM];
//   __shared__ scalar_t Bs[BLOCK_DIM][BLOCK_DIM];

//   for (int k = 0; k < (N + BLOCK_DIM - 1) / BLOCK_DIM; ++k) {
//     if (row < M && k * BLOCK_DIM + tx < N)
//       As[ty][tx] = a[row * N + k * BLOCK_DIM + tx];
//     else
//       As[ty][tx] = 0.0;

//     if (col < P && k * BLOCK_DIM + ty < N)
//       Bs[ty][tx] = b[(k * BLOCK_DIM + ty) * P + col];
//     else
//       Bs[ty][tx] = 0.0;

//     __syncthreads();

//     for (int n = 0; n < BLOCK_DIM; ++n)
//       sum += As[ty][n] * Bs[n][tx];

//     __syncthreads();
//   }

//   if (row < M && col < P)
//     out[row * P + col] = sum;
// }

// #define THREAD_DIM 2


// __global__ void Block2DMatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N, uint32_t P)
// {
//   int bx = blockIdx.x;
//   int by = blockIdx.y;
//   int tidx = threadIdx.x;

//   __shared__ scalar_t As[BLOCK_DIM][BLOCK_DIM];
//   __shared__ scalar_t Bs[BLOCK_DIM][BLOCK_DIM];

//   int block_row = by * BLOCK_DIM;
//   int block_col = bx * BLOCK_DIM;

//   int load_row = tidx / BLOCK_DIM;
//   int load_col = tidx % BLOCK_DIM;

//   int compute_row = tidx / (BLOCK_DIM / THREAD_DIM) * THREAD_DIM;
//   int compute_col = tidx % (BLOCK_DIM / THREAD_DIM) * THREAD_DIM;

//   a = a + block_row * N;
//   b = b + block_col;
//   out = out + block_row * P + block_col;
  
//   scalar_t result[THREAD_DIM * THREAD_DIM] = {0.0};
  
//   for(size_t k = 0; k < (N + BLOCK_DIM - 1) / BLOCK_DIM; k++) {
//     //load As
//     for(size_t i = 0; i < BLOCK_DIM * BLOCK_DIM / BASE_THREAD_NUM; i++) {
//       int share_row = load_row + i * (BASE_THREAD_NUM / BLOCK_DIM);
//       int share_col = load_col;
//       if((block_row + share_row) < M && (k * BLOCK_DIM + share_col) < N)
//         As[share_row][share_col] = a[share_row * N + share_col];
//       else 
//         As[share_row][share_col] = 0.0;
//     }

//     //load Bs
//     for(size_t i = 0; i < BLOCK_DIM * BLOCK_DIM / BASE_THREAD_NUM; i++) {
//       int share_row = load_row + i * (BASE_THREAD_NUM / BLOCK_DIM);
//       int share_col = load_col;
//       if((k * BLOCK_DIM + share_row) < N && (block_col + share_col) < P)
//         Bs[share_row][share_col] = b[share_row * P + share_col];
//       else 
//         Bs[share_row][share_col] = 0.0;
//     }
    
//     __syncthreads();

//     //compute
//     for(int m = 0; m < BLOCK_DIM; m++) {
//       for(int i = 0; i < THREAD_DIM; i++) {
//         for(int j = 0; j < THREAD_DIM; j++) {
//           result[i * THREAD_DIM + j] += As[compute_row + i][m] * Bs[m][compute_col + j];
//         }
//       }
//     }
//     __syncthreads();
    
//     a += BLOCK_DIM;
//     b += BLOCK_DIM * P;
//   }

//   for(int i = 0; i < THREAD_DIM; i++) {
//     for(int j = 0; j < THREAD_DIM; j++) {
//       if((compute_row + block_row + i) < M && (compute_col + block_col + j) < P) {
//         out[(compute_row + i) * P + compute_col + j] = result[i * 2 + j];
//       }
//     }
//   }
// }

// void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
//             uint32_t P) {
//   /**
//    * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
//    * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
//    * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
//    * over (i,j) entries in the output array.  However, to really get the full benefit of this
//    * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
//    * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
//    * the CPU backend, here you should implement a single function that works across all size
//    * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
//    * implementations, this function here will largely just set up the kernel call, and you should
//    * implement the logic in a separate MatmulKernel() call.
//    * 
//    *
//    * Args:
//    *   a: compact 2D array of size m x n
//    *   b: comapct 2D array of size n x p
//    *   out: compact 2D array of size m x p to write the output to
//    *   M: rows of a / out
//    *   N: columns of a / rows of b
//    *   P: columns of b / out
//    */

//   /// BEGIN SOLUTION
//   // CudaDims dim = CudaOneDim(out->size);
//   // NaiveMatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);

//   // dim3 block(BLOCK_DIM, BLOCK_DIM);
//   // dim3 grid((P + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);
//   // ShareMemoryMatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);

//   dim3 block(BASE_THREAD_NUM);
//   dim3 grid((P + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);
//   Block2DMatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
//   /// END SOLUTION
// }

// ////////////////////////////////////////////////////////////////////////////////
// // Max and sum reductions
// ////////////////////////////////////////////////////////////////////////////////
// __global__ void ReduceMaxKernel(const scalar_t *a, scalar_t *out, size_t reduce_size, size_t size)
// {
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if(gid >= size) return;
//   auto a_ptr = a + reduce_size * gid;
//   auto out_ptr = out + gid;
//   scalar_t max_num = a_ptr[0];
//   for(size_t i = 0; i < reduce_size; i++) {
//     max_num = max(max_num, a_ptr[i]);
//   }

//   *out_ptr = max_num;
// }

// void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
//   /**
//    * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
//    * for simplicity you can perform each reduction in a single CUDA thread.
//    * 
//    * Args:
//    *   a: compact array of size a.size = out.size * reduce_size to reduce over
//    *   out: compact array to write into
//    *   redice_size: size of the dimension to reduce over
//    */
//   /// BEGIN SOLUTION
//   CudaDims dim = CudaOneDim(out->size);
  
//   ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);

//   /// END SOLUTION
// }

// __global__ void ReduceSumKernel(const scalar_t *a, scalar_t *out, size_t reduce_size, size_t size)
// {
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if(gid >= size) return;

//   auto a_ptr = a + reduce_size * gid;
//   auto out_ptr = out + gid;

//   scalar_t sum = 0; 
//   for(size_t i = 0; i < reduce_size; i++) {
//     sum += a_ptr[i];
//   }

//   *out_ptr = sum;
// }

// void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
//   /**
//    * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
//    * can perform each reduction in a single CUDA thread.
//    * 
//    * Args:
//    *   a: compact array of size a.size = out.size * reduce_size to reduce over
//    *   out: compact array to write into
//    *   redice_size: size of the dimension to reduce over
//    */
//   /// BEGIN SOLUTION
//   CudaDims dim = CudaOneDim(out->size);
//   ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);

//   /// END SOLUTION
// }

#ifdef NEEDLE_ENABLE_FLASHATTN_STUB

template <typename GLayoutQ, typename GLayoutK, typename GLayoutV, typename GLayoutO,
          typename SLayoutQ, typename SLayoutK, typename SLayoutV, typename SLayoutO,
          typename TileG2SQ, typename TileG2SK, typename TileG2SV,
          typename MMA1, typename MMA2>
__global__ void flash_attention_kernel(const scalar_t* q, const scalar_t* k, const scalar_t* v,
                                       scalar_t* o, float dropout, bool causal,
                                       GLayoutQ gQL, GLayoutK gKL, GLayoutV gVL, GLayoutO gOL,
                                       SLayoutQ sQL, SLayoutK sKL, SLayoutV sVL, SLayoutO sOL,
                                       TileG2SQ g2s_copyQ, TileG2SK g2s_copyK, TileG2SV g2s_copyV,
                                       MMA1 mma1, MMA2 mma2)
{
  using namespace cute;

  extern __shared__ __align__(sizeof(float)) unsigned char smem[];

  auto align_up = [](size_t offset, size_t alignment) {
    return ((offset + alignment - 1) / alignment) * alignment;
  };

  size_t smem_offset = 0;
  scalar_t* qShmem = reinterpret_cast<scalar_t*>(smem + smem_offset);
  smem_offset += size(sQL) * sizeof(scalar_t);
  scalar_t* kShmem = reinterpret_cast<scalar_t*>(smem + smem_offset);
  smem_offset += size(sKL) * sizeof(scalar_t);
  scalar_t* vShmem = reinterpret_cast<scalar_t*>(smem + smem_offset);
  smem_offset += size(sVL) * sizeof(scalar_t);
  smem_offset = align_up(smem_offset, alignof(float));
  float* oShmem = reinterpret_cast<float*>(smem + smem_offset);
  smem_offset += size(sOL) * sizeof(float);
  float* pShmem = reinterpret_cast<float*>(smem + smem_offset);
  smem_offset += size<0>(sQL) * size<0>(sKL) * sizeof(float);
  float* row_maxShmem = reinterpret_cast<float*>(smem + smem_offset);
  smem_offset += size<0>(sQL) * sizeof(float);
  float* row_sumShmem = reinterpret_cast<float*>(smem + smem_offset);

  for (int i = threadIdx.x; i < size(sOL); i += blockDim.x) {
    oShmem[i] = 0.0f;
  }
  __syncthreads();

  Tensor Q = make_tensor(make_gmem_ptr<scalar_t>(q), gQL);   // (batch_size, num_heads, q_len, head_dim)
  Tensor K = make_tensor(make_gmem_ptr<scalar_t>(k), gKL);   // (batch_size, num_heads, kv_len, head_dim)
  Tensor V = make_tensor(make_gmem_ptr<scalar_t>(v), gVL);   // (batch_size, num_heads, kv_len, head_dim)
  Tensor O = make_tensor(make_gmem_ptr<scalar_t>(o), gOL);   // (batch_size, num_heads, q_len, head_dim)

  int bid = blockIdx.x;
  int num_tiles = size<2>(gQL) / size<0>(sQL);
  int seq_id   = bid % num_tiles;
  int head_id  = (bid / num_tiles) % size<1>(gQL);
  int batch_id = bid / (num_tiles * size<1>(gQL));

  auto gQ = local_tile(Q(batch_id, head_id, _, _),
                       make_tile(size<0>(sQL), size<1>(sQL)),
                       make_coord(seq_id, _0{}));               // (bN, head_dim)

  // auto gK = local_tile(K(batch_id, head_id, _, _),
  //                      make_tile(size<0>(sKL), size<1>(sKL)),
  //                      make_coord(_, _));                    // (bK, head_dim, seq_len, 1)

  // auto gV = local_tile(V(batch_id, head_id, _, _),
  //                      make_tile(size<0>(sVL), size<1>(sVL)),
  //                      make_coord(_, _));                    // (bK, head_dim, seq_len, 1)

  auto gO = local_tile(O(batch_id, head_id, _, _),
                       make_tile(size<0>(sOL), size<1>(sOL)),
                       make_coord(seq_id, _0{}));               // (bN, head_dim)

  Tensor sQ = make_tensor(make_smem_ptr<scalar_t>(qShmem), sQL);   // (bN, head_dim)
  Tensor sK = make_tensor(make_smem_ptr<scalar_t>(kShmem), sKL);   // (bK, head_dim)
  Tensor sV = make_tensor(make_smem_ptr<scalar_t>(vShmem), sVL);   // (bK, head_dim)
  Tensor sV_trans  = make_tensor(make_smem_ptr<scalar_t>(vShmem), 
                                make_layout(make_shape(size<1>(sVL), size<0>(sVL)), 
                                make_stride(stride<1>(sVL), stride<0>(sVL))));   // (head_dim, bK)
  Tensor sO = make_tensor(make_smem_ptr<float>(oShmem), sOL);   // (bN, head_dim)

  Tensor sP = make_tensor(make_smem_ptr<float>(pShmem),
                          make_shape(size<0>(sQL), size<0>(sKL)));  // (bN, bK)

  Tensor row_max = make_tensor(make_smem_ptr<float>(row_maxShmem),
                               make_shape(size<0>(sQL)));           // (bN)

  Tensor row_sum = make_tensor(make_smem_ptr<float>(row_sumShmem),
                               make_shape(size<0>(sQL)));           // (bN)

  // 初始化 row_max / row_sum
  if (threadIdx.x < size<0>(row_max)) {
    row_max(threadIdx.x) = -FLT_MAX;
    row_sum(threadIdx.x) = 0.0f;
  }
  __syncthreads();

  // load gQ to sQ
  auto g2s_thr_copyQ = g2s_copyQ.get_slice(threadIdx.x);
  auto tQgQ = g2s_thr_copyQ.partition_S(gQ);   // (CPY, CPY_M, CPY_N)
  auto tQsQ = g2s_thr_copyQ.partition_D(sQ);   // (CPY, CPY_M, CPY_N)

  // load gK to sK
  auto g2s_thr_copyK = g2s_copyK.get_slice(threadIdx.x);
  //auto tKgK = g2s_thr_copyK.partition_S(gK);   // (CPY, CPY_M, CPY_N, seq_len)
  auto tKsK = g2s_thr_copyK.partition_D(sK);   // (CPY, CPY_M, CPY_N)

  // load gV to sV
  auto g2s_thr_copyV = g2s_copyV.get_slice(threadIdx.x);
  //auto tVgV = g2s_thr_copyV.partition_S(gV);   // (CPY, CPY_M, CPY_N, seq_len)
  auto tVsV = g2s_thr_copyV.partition_D(sV);   // (CPY, CPY_M, CPY_N)

  // mma1 and alloc register
  auto thr_mma1 = mma1.get_slice(threadIdx.x);
  auto mma1sQ = thr_mma1.partition_A(sQ);   // (MMA, Mma_M, Mma_K)
  auto mma1sK = thr_mma1.partition_B(sK);
  auto mma1sP = thr_mma1.partition_C(sP);   // (MMA, Mma_M, Mma_N)
  auto mma1rQ = thr_mma1.make_fragment_A(mma1sQ);   // (MMA, Mma_M, Mma_K)
  auto mma1rK = thr_mma1.make_fragment_B(mma1sK);
  auto mma1rP = thr_mma1.make_fragment_C(mma1sP);   // (MMA, Mma_M, Mma_N)
  //clear(mma1rP);

  // mma2 and alloc register
  auto thr_mma2 = mma2.get_slice(threadIdx.x);
  auto mma2sP = thr_mma2.partition_A(sP);   // (MMA, Mma_M, Mma_K)
  auto mma2sV = thr_mma2.partition_B(sV_trans);
  auto mma2sO = thr_mma2.partition_C(sO);   // (MMA, Mma_M, Mma_N)
  auto mma2rP = make_fragment_like<scalar_t>(mma2sP);
  auto mma2rV = thr_mma2.make_fragment_B(mma2sV);
  auto mma2rO = thr_mma2.make_fragment_C(mma2sO);
  //clear(mma2rO);

  // load gQ to sQ and load sQ to register
  copy(g2s_copyQ, tQgQ, tQsQ);
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();
  copy(mma1sQ, mma1rQ);
  float sm_scale = rsqrtf(size<1>(sQ));  // 1 / sqrt(head_dim)

  for (size_t t = 0; t < size(mma1rQ); ++t) {
    //mma1rQ(t) = to_tf32(mma1rQ(t)* sm_scale);
    mma1rQ(t) = mma1rQ(t) * sm_scale;
  }

  for (size_t i = 0; i < size<2>(gKL) / size<0>(sKL); ++i) {
    clear(mma1rP);
    // load gK to sK and load gV to sV
    auto gK = local_tile(K(batch_id, head_id, _, _),
                       make_tile(size<0>(sKL), size<1>(sKL)),
                       make_coord(i, 0));                    // (bK, head_dim)
    auto tKgK = g2s_thr_copyK.partition_S(gK);
    copy(g2s_copyK, tKgK, tKsK);
    cp_async_fence();

    auto gV = local_tile(V(batch_id, head_id, _, _),
                       make_tile(size<0>(sVL), size<1>(sVL)),
                       make_coord(i, 0));                    // (bK, head_dim)
    auto tVgV = g2s_thr_copyV.partition_S(gV);
    copy(g2s_copyV, tVgV, tVsV);
    cp_async_fence();

    cp_async_wait<1>();
    __syncthreads();
    // load sK to register
    copy(mma1sK, mma1rK);
    // for (size_t j = 0; j < size(mma1rK); ++j) {
    //   mma1rK(j) = to_tf32(mma1rK(j));
    // }
    gemm(mma1, mma1rQ, mma1rK, mma1rP);

    copy(mma1rP, mma1sP);
    __syncthreads();

    int row = threadIdx.x;
    if (row < size<0>(sP)) {
      float row_max_new = row_max(row);

      for (int n = 0; n < size<1>(sP); ++n) {
        row_max_new = max(row_max_new, sP(row, n));
      }

      float sum = 0.0f;
      float alpha = exp(row_max(row) - row_max_new);

      for (int d = 0; d < size<1>(sO); ++d) {
        sO(row, d) = sO(row, d) * alpha;
      }

      for (int n = 0; n < size<1>(sP); ++n) {
        sP(row, n) = exp(sP(row, n) - row_max_new);
        sum += sP(row, n);
        //sO(row, n) = sO(row, n) * offset;
      }

      row_sum(row) = row_sum(row) * alpha + sum;
      row_max(row) = row_max_new;
    }

    cp_async_wait<0>();
    __syncthreads();

    copy(mma2sP, mma2rP);
    copy(mma2sV, mma2rV);
    // for (int i = 0; i < size(mma2rP); ++i) {
    //   mma2rP(i) = to_tf32(mma2rP(i));
    // }
    // for (int i = 0; i < size(mma2rV); ++i) {
    //   mma2rV(i) = to_tf32(mma2rV(i));
    // }

    copy(mma2sO, mma2rO);

    gemm(mma2, mma2rP, mma2rV, mma2rO);
    copy(mma2rO, mma2sO);
    __syncthreads();
  }
  // final scale: O /= row_sum
  if (threadIdx.x < size<0>(sO)) {
    int row = threadIdx.x;
    float denom = row_sum(row);
    if (denom > 0.0f) {
      for (int d = 0; d < size<1>(sO); ++d) {
        sO(row, d) = sO(row, d) / denom;
      }
    }
  }
  __syncthreads();

for (int i = threadIdx.x; i < size(sO); i += blockDim.x) {
    gO(i) = sO(i);
}
  return;
}

void FlashAttentionForward(const CudaArray& q, const CudaArray& k, const CudaArray& v,
                          CudaArray* out, size_t batch_size, size_t num_heads,
                          size_t q_len, size_t kv_len,
                          size_t head_dim, float dropout, bool causal) {
  (void)sizeof(cutlass::Status);

  const size_t q_size = batch_size * num_heads * q_len * head_dim;
  const size_t kv_size = batch_size * num_heads * kv_len * head_dim;
  const size_t out_size = q_size;

  if (q.size != q_size) throw std::runtime_error("flash attention q tensor size does not match the provided metadata");
  if (k.size != kv_size) throw std::runtime_error("flash attention k tensor size does not match the provided metadata");
  if (v.size != kv_size) throw std::runtime_error("flash attention v tensor size does not match the provided metadata");
  if (out->size != out_size) throw std::runtime_error("flash attention output tensor size does not match the provided metadata");

  std::ostringstream msg;
  msg << "flash attention kernel"
      << " (batch_size=" << batch_size
      << ", num_heads=" << num_heads
      << ", q_len=" << q_len
      << ", kv_len=" << kv_len
      << ", head_dim=" << head_dim
      << ", dropout=" << dropout
      << ", causal=" << std::boolalpha << causal << ")";
  
  std::cout << msg.str() << std::endl;

  using namespace cute;
  auto gQ = make_layout(make_shape(batch_size, num_heads, q_len, _64{}), LayoutRight{});
  auto gK = make_layout(make_shape(batch_size, num_heads, kv_len,  _64{}), LayoutRight{});
  auto gV = make_layout(make_shape(batch_size, num_heads, kv_len, _64{}), LayoutRight{});
  auto gO = make_layout(make_shape(batch_size, num_heads, q_len, _64{}), LayoutRight{});

  auto bN = Int<16>{};
  auto bK = Int<16>{};

  auto sQ = make_layout(make_shape(bN,  _64{}), LayoutRight{});
  auto sK = make_layout(make_shape(bK,  _64{}), LayoutRight{});
  auto sV = make_layout(make_shape(bK,  _64{}), LayoutRight{});
  auto sO = make_layout(make_shape(bN,  _64{}), LayoutRight{});

  //copy gobal to share memory
  auto g2s_copyQ = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, scalar_t>{}, 
                                  Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                  Layout<Shape<_1, _8>>{});
  auto g2s_copyK = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, scalar_t>{}, 
                                  Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                  Layout<Shape<_1, _8>>{});
  auto g2s_copyV = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, scalar_t>{}, 
                                  Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                  Layout<Shape<_1, _8>>{});

  //copy share to register using navie copy 
  
  //mma1
  auto mma1 = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
                                Layout<Shape<_1, _2, _2>>{},
                                Tile<_16, _16, _32>{});
  auto mma2 = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
                                Layout<Shape<_1, _4, _1>>{},
                                Tile<_16, _32, _16>{});
  // auto mma1 = make_tiled_mma(UniversalFMA<scalar_t, scalar_t, float>{},
  //                               Layout<Shape<_16, _8, _1>>{});
  // auto mma2 = make_tiled_mma(UniversalFMA<scalar_t, scalar_t, float>{},
  //                               Layout<Shape<_16, _8, _1>>{});
  size_t smem_elems_half = (bN + bK + bK) * head_dim;
  size_t smem_elems_float = bN * 2 + bN * bK + bN * head_dim; // sQ, sK, sV, sO, sP, row_max, row_sum
  size_t smem_half_bytes = smem_elems_half * sizeof(scalar_t);
  size_t smem_bytes = ((smem_half_bytes + alignof(float) - 1) / alignof(float)) * alignof(float)
                    + smem_elems_float * sizeof(float);
  // parition 
  dim3 block(128);
  // assum q_len % 16 == 0
  dim3 grid(batch_size * num_heads * q_len / bN);

  flash_attention_kernel<<<grid, block, smem_bytes>>>(q.ptr, k.ptr, v.ptr, out->ptr, dropout, causal,
                                          gQ, gK, gV, gO,
                                          sQ, sK, sV, sO,
                                          g2s_copyQ, g2s_copyK, g2s_copyV,
                                          mma1, mma2);


  return;
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
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
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
  m.def("flash_attention_forward", FlashAttentionForward);
#else
  m.attr("__has_flash_attention_stub__") = false;
#endif
}
