#include <cstdint>
#include <cfloat>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

#include <cassert>

#ifdef NEEDLE_ENABLE_FLASHATTN_STUB
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cute/util/print.hpp>
#include <cute/util/print_latex.hpp>
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


#ifdef NEEDLE_ENABLE_FLASHATTN_STUB

using namespace cute;

void CudaCheck(cudaError_t err, const char* context) {
  if (err != cudaSuccess) {
    std::ostringstream msg;
    msg << context << ": " << cudaGetErrorString(err);
    throw std::runtime_error(msg.str());
  }
}

template <typename Engine, typename Layout, typename EngineOut>
CUTLASS_DEVICE void convert_type_out(Tensor<Engine, Layout> const &tensor, Tensor<EngineOut, Layout> &out) {
    // Somehow if we allocate out inside this function and return it, e2e is slower and the output can be wrong.
    using From_type = typename Engine::value_type;
    using To_type = typename EngineOut::value_type;
    static constexpr int FragmentSize = (sizeof(From_type) > sizeof(To_type)) ? (sizeof(From_type) / sizeof(To_type)) : (sizeof(To_type) / sizeof(From_type));    
    static_assert(CUTE_STATIC_V(size(tensor)) % FragmentSize == 0, "Fragment size does not vectorize properly");
    Tensor frag = recast<cutlass::Array<From_type, FragmentSize> const>(tensor);
    Tensor out_frg = recast<cutlass::Array<To_type, FragmentSize>>(out);
    static_assert(size(frag) == size(out_frg));
    cutlass::NumericArrayConverter<To_type, From_type, FragmentSize> convert_op;
    #pragma unroll
    for (int i = 0; i < size(frag); ++i) { out_frg[i] = convert_op(frag[i]); }
}

template<bool isFirst, typename FragmentP, typename FragmentO>
CUTLASS_DEVICE void online_softmax(FragmentP& rP, float* row_max, float* row_sum, FragmentO& rO) 
{
  float row_max_new[2] = {row_max[0], row_max[1]};

  //get local thread row max
  for(size_t i = 0; i < size<2>(rP); i++) {
    for(size_t j = 0; j < size<0, 0>(rP); j++) {
      row_max_new[0] = rP(make_coord(j, 0), 0, i) > row_max_new[0] ? rP(make_coord(j, 0), 0, i) : row_max_new[0];
      row_max_new[1] = rP(make_coord(j, 1), 0, i) > row_max_new[1] ? rP(make_coord(j, 1), 0, i) : row_max_new[1];
    }
  }
  //shuffle  every thread get the max value of the row
  unsigned mask = unsigned(-1);
  row_max_new[0] = fmaxf(row_max_new[0], __shfl_xor_sync(mask, row_max_new[0], 1, 4));
  row_max_new[0] = fmaxf(row_max_new[0], __shfl_xor_sync(mask, row_max_new[0], 2, 4));
  row_max_new[1] = fmaxf(row_max_new[1], __shfl_xor_sync(mask, row_max_new[1], 1, 4));
  row_max_new[1] = fmaxf(row_max_new[1], __shfl_xor_sync(mask, row_max_new[1], 2, 4));

  float alph[2];
  alph[0] = expf(row_max[0] - row_max_new[0]);
  alph[1] = expf(row_max[1] - row_max_new[1]);

  //get local thread sum
  float row_sum_new[2] = {0.0f, 0.0f};
  for(size_t i = 0; i < size<2>(rP); i++) {
    for(size_t j = 0; j < size<0, 0>(rP); j++) {
      rP(make_coord(j, 0), 0, i) = expf(rP(make_coord(j, 0), 0, i) - row_max_new[0]);
      rP(make_coord(j, 1), 0, i) = expf(rP(make_coord(j, 1), 0, i) - row_max_new[1]);
      row_sum_new[0] += rP(make_coord(j, 0), 0, i);
      row_sum_new[1] += rP(make_coord(j, 1), 0, i);

    }
  }
  //shuffle to get the sum of the row
  row_sum_new[0] += __shfl_xor_sync(mask, row_sum_new[0], 1, 4);
  row_sum_new[0] += __shfl_xor_sync(mask, row_sum_new[0], 2, 4);
  row_sum_new[1] += __shfl_xor_sync(mask, row_sum_new[1], 1, 4);
  row_sum_new[1] += __shfl_xor_sync(mask, row_sum_new[1], 2, 4);

  //update rO
  if(!isFirst) {
    for(size_t i = 0; i < size<2>(rO); i++) {
      for(size_t j = 0; j < size<0, 0>(rO); j++) {
        rO(make_coord(j, 0), 0, i) = rO(make_coord(j, 0), 0, i) * alph[0];
        rO(make_coord(j, 1), 0, i) = rO(make_coord(j, 1), 0, i) * alph[1];
      }
    }
  }


  //update row_max and row_sum
  if(!isFirst) {
    row_sum[0] = row_sum_new[0] + row_sum[0] * alph[0];
    row_sum[1] = row_sum_new[1] + row_sum[1] * alph[1];
  } else {
    row_sum[0] = row_sum_new[0];
    row_sum[1] = row_sum_new[1];
  }

  row_max[0] = row_max_new[0];
  row_max[1] = row_max_new[1];

  return;

}
template <typename GLayoutQ, typename GLayoutK, typename GLayoutV, typename GLayoutO,
          typename SLayoutQ, typename SLayoutK, typename SLayoutV, typename SLayoutO, typename SLayoutVTrans_t,
          typename TileG2SQ, typename TileG2SK, typename TileG2SV,
          typename AtomS2RQ, typename AtomS2RK, typename AtomS2RV,
          typename MMA1, typename MMA2, 
          typename TiledS2GO>
__global__ void flash_attention_kernel(const scalar_t* q, const scalar_t* k, const scalar_t* v,
                                       scalar_t* o, float dropout, bool causal,
                                       GLayoutQ gQL, GLayoutK gKL, GLayoutV gVL, GLayoutO gOL,
                                       SLayoutQ sQL, SLayoutK sKL, SLayoutV sVL, SLayoutO sOL, SLayoutVTrans_t sVTransL,
                                       TileG2SQ g2s_copyQ, TileG2SK g2s_copyK, TileG2SV g2s_copyV,
                                       AtomS2RQ s2r_copyQ_atom, AtomS2RK s2r_copyK_atom, AtomS2RV s2r_copyV_atom,
                                       MMA1 mma1, MMA2 mma2, 
                                       TiledS2GO s2g_copyO)
{

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
  
  float row_max[2] = {-FLT_MAX, -FLT_MAX};
  float row_sum[2] = {0.0f, 0.0f};

  

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


  auto gO = local_tile(O(batch_id, head_id, _, _),
                       make_tile(size<0>(sOL), size<1>(sOL)),
                       make_coord(seq_id, _0{}));               // (bN, head_dim)

  Tensor sQ = make_tensor(make_smem_ptr<scalar_t>(qShmem), sQL);   // (bN, head_dim)
  Tensor sK = make_tensor(make_smem_ptr<scalar_t>(kShmem), sKL);   // (bK, head_dim)
  Tensor sV = make_tensor(make_smem_ptr<scalar_t>(vShmem), sVL);   // (bK, head_dim)
  Tensor sV_trans  = make_tensor(make_smem_ptr<scalar_t>(vShmem), sVTransL);   // (head_dim, bK)
  Tensor sO = make_tensor(make_smem_ptr<scalar_t>(oShmem), sOL);   // (bN, head_dim)

  // Tensor sP = make_tensor(make_smem_ptr<float>(pShmem),
  //                         make_shape(size<0>(sQL), size<0>(sKL)));  // (bN, bK)
  auto sP = make_coord_tensor(make_layout(make_shape(size<0>(sQL), size<0>(sKL))));  // (bN, bK)



  // load gQ to sQ
  auto g2s_thr_copyQ = g2s_copyQ.get_slice(threadIdx.x);
  auto tQgQ = g2s_thr_copyQ.partition_S(gQ);   // (CPY, CPY_M, CPY_N)
  auto tQsQ = g2s_thr_copyQ.partition_D(sQ);   // (CPY, CPY_M, CPY_N)

  // load gK to sK
  auto g2s_thr_copyK = g2s_copyK.get_slice(threadIdx.x);
  auto tKsK = g2s_thr_copyK.partition_D(sK);   // (CPY, CPY_M, CPY_N)

  // load gV to sV
  auto g2s_thr_copyV = g2s_copyV.get_slice(threadIdx.x);
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
  // auto mma2rP = thr_mma2.make_fragment_A(mma2sP);   // (MMA, Mma_M, Mma_K)
  // auto mma2rP_float = make_fragment_like<float>(mma2sP);   // (MMA, Mma_M, Mma_K)
  auto mma2rP = make_fragment_like<scalar_t>(mma2sP);
  auto mma2rV = thr_mma2.make_fragment_B(mma2sV);
  auto mma2rO = thr_mma2.make_fragment_C(mma2sO);
  auto mma2gO = thr_mma2.partition_C(gO);
  // auto mma2rO_half = make_fragment_like<scalar_t>(mma2sO);
  clear(mma2rO);

  // load sQ to rQ
  auto s2r_tiled_copyQ = make_tiled_copy_A(s2r_copyQ_atom, mma1);
  auto s2r_thr_copyQ = s2r_tiled_copyQ.get_slice(threadIdx.x);
  auto tQsQ_s2r = s2r_thr_copyQ.partition_S(sQ);
  auto tQrQ_s2r = s2r_thr_copyQ.retile_D(mma1rQ);
  //load sK to rK
  auto s2r_tiled_copyK = make_tiled_copy_B(s2r_copyK_atom, mma1);
  auto s2r_thr_copyK = s2r_tiled_copyK.get_slice(threadIdx.x);
  auto tKsK_s2r = s2r_thr_copyK.partition_S(sK);
  auto tKrK_s2r = s2r_thr_copyK.retile_D(mma1rK);
  //load sV to rV
  auto s2r_tiled_copyV = make_tiled_copy_B(s2r_copyV_atom, mma2);
  auto s2r_thr_copyV = s2r_tiled_copyV.get_slice(threadIdx.x);
  auto tVsV_s2r = s2r_thr_copyV.partition_S(sV_trans);
  auto tVrV_s2r = s2r_thr_copyV.retile_D(mma2rV);

  //load sO to gO
  auto g2s_thr_copyO = s2g_copyO.get_slice(threadIdx.x);
  auto tOsO = g2s_thr_copyO.partition_S(sO);
  auto tOgO = g2s_thr_copyO.partition_D(gO);

  if(thread0()) {
    // print(mma1sP); print("\n");
    // print(mma1rP); print("\n");
    // print(mma2sP); print("\n");
    // // print(mma2rP_float); print("\n");
    // print(mma2rP); print("\n");
    // print(mma2sO); print("\n");
    // print(mma2rO); print("\n");
    // print_latex(mma1); print("\n");
    // print_latex(mma2); print("\n");
    // print_latex(s2r_tiled_copyQ); print("\n");
    // print_latex(s2r_tiled_copyK); print("\n");
    // print_latex(s2r_tiled_copyV); print("\n");
    print(mma2gO); print("\n");
    print(mma2sO); print("\n");
    print(mma2rO); print("\n");
  }

  // load gQ to sQ and load sQ to register
  copy(g2s_copyQ, tQgQ, tQsQ);
  cp_async_fence();

  // load gK_0 to sK 
  auto gK = local_tile(K(batch_id, head_id, _, _),
                       make_tile(size<0>(sKL), size<1>(sKL)),
                       make_coord(0, 0)); // (bK, head_dim)
  auto tKgK = g2s_thr_copyK.partition_S(gK);
  copy(g2s_copyK, tKgK, tKsK);  
  cp_async_fence();
  cp_async_wait<1>();
  __syncthreads();  

  //copy sQ to register 
  copy(s2r_tiled_copyQ, tQsQ_s2r, tQrQ_s2r);
  float sm_scale = rsqrtf(size<1>(sQ));  // 1 / sqrt(head_dim)

  #pragma unroll
  for (size_t t = 0; t < size(mma1rQ); ++t) {
    mma1rQ(t) = mma1rQ(t) * sm_scale;
  }

  

  for (size_t i = 0; i < size<2>(gKL) / size<0>(sKL); ++i) {
    clear(mma1rP);

    auto gV = local_tile(V(batch_id, head_id, _, _),
                       make_tile(size<0>(sVL), size<1>(sVL)),
                       make_coord(i, 0));                    // (bK, head_dim)
    auto tVgV = g2s_thr_copyV.partition_S(gV);
    copy(g2s_copyV, tVgV, tVsV);
    cp_async_fence();

    cp_async_wait<1>();
    __syncthreads();
    // load sK to register
    copy(s2r_tiled_copyK, tKsK_s2r, tKrK_s2r);
    //__syncthreads();

    //load gV_{i+1} to sV
    if(i != size<2>(gKL) / size<0>(sKL) - 1) {
      gK = local_tile(K(batch_id, head_id, _, _),
                         make_tile(size<0>(sKL), size<1>(sKL)),
                         make_coord(i + 1, 0));                    // (bK, head_dim)
      tKgK = g2s_thr_copyK.partition_S(gK);
      copy(g2s_copyK, tKgK, tKsK);
      cp_async_fence();
    }

    gemm(mma1, mma1rQ, mma1rK, mma1rP);


    if(i == 0) {
      online_softmax<true>(mma1rP, row_max, row_sum, mma2rO);
    } else {
      online_softmax<false>(mma1rP, row_max, row_sum, mma2rO);
    }
    convert_type_out(mma1rP, mma2rP);
    
    if(i != size<2>(gKL) / size<0>(sKL) - 1) {
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    //copy sV to register 
    copy(s2r_tiled_copyV, tVsV_s2r, tVrV_s2r);


    gemm(mma2, mma2rP, mma2rV, mma2rO);
    //copy(mma2rO, mma2sO);
    __syncthreads();
  }

  float row_sum_inv[2];
  row_sum_inv[0] = 1.0f / row_sum[0];
  row_sum_inv[1] = 1.0f / row_sum[1];
  for(size_t i = 0; i < size<2>(mma2rO); i++) {
    for(size_t j = 0; j < size<0, 0>(mma2rO); j++) {
      mma2rO(make_coord(j, 0), 0, i) = mma2rO(make_coord(j, 0), 0, i) *row_sum_inv[0];
      mma2rO(make_coord(j, 1), 0, i) = mma2rO(make_coord(j, 1), 0, i) *row_sum_inv[1];
    }
  }
  
  copy(mma2rO, mma2sO);
  __syncthreads();

  copy(s2g_copyO, tOsO, tOgO);
  return;
}

void LaunchFlashAttentionForward(const CudaArray& q, const CudaArray& k, const CudaArray& v,
                                 CudaArray* out, size_t batch_size, size_t num_heads,
                                 size_t q_len, size_t kv_len,
                                 size_t head_dim, float dropout, bool causal,
                                 cudaStream_t stream) {
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
  
  // std::cout << msg.str() << std::endl;

  using namespace cute;
  auto gQ = make_layout(make_shape(batch_size, num_heads, q_len, _64{}), LayoutRight{});
  auto gK = make_layout(make_shape(batch_size, num_heads, kv_len,  _64{}), LayoutRight{});
  auto gV = make_layout(make_shape(batch_size, num_heads, kv_len, _64{}), LayoutRight{});
  auto gO = make_layout(make_shape(batch_size, num_heads, q_len, _64{}), LayoutRight{});

  auto bN = Int<64>{};
  auto bK = Int<64>{};

  //auto sQ = make_layout(make_shape(bN,  _64{}), LayoutRight{});
  auto sQ = composition(Swizzle<3,3,3>{}, make_layout(make_shape(bN,  _64{}), LayoutRight{}));
  //auto sK = make_layout(make_shape(bK,  _64{}), LayoutRight{});
  auto sK = composition(Swizzle<3,3,3>{}, make_layout(make_shape(bK,  _64{}), LayoutRight{}));
  //auto sV = make_layout(make_shape(bK,  _64{}), LayoutRight{});
  auto sV = composition(Swizzle<3,3,3>{}, make_layout(make_shape(bK, _64{}), LayoutRight{}));
  auto sV_trans = composition(Swizzle<3,3,3>{}, make_layout(make_shape(_64{}, bK), LayoutLeft{}));
  //auto sV_trans = make_layout(make_shape(_64{}, bK), LayoutLeft{});
  auto sO = make_layout(make_shape(bN,  _64{}), LayoutRight{});

  //copy gobal to share memory
  auto g2s_copyQ = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, scalar_t>{}, 
                                  Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                  Layout<Shape<_1, _8>>{});
  auto g2s_copyK = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, scalar_t>{}, 
                                  Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                  Layout<Shape<_1, _8>>{});
  auto g2s_copyV = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, scalar_t>{}, 
                                  Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                  Layout<Shape<_1, _8>>{});

  //copy share to register using ldmatrix copy 
  auto s2r_copyQ_atom = Copy_Atom<SM75_U32x1_LDSM_N, scalar_t>{};
  auto s2r_copyK_atom = Copy_Atom<SM75_U32x1_LDSM_N, scalar_t>{};
  auto s2r_copyV_atom = Copy_Atom<SM75_U16x2_LDSM_T, scalar_t>{};
  //mma1
  auto mma1 = make_tiled_mma(MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>{},
                                Layout<Shape<_4, _1, _1>>{},
                                Tile<_64, _64, _8>{});
  auto mma2 = make_tiled_mma(MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>{},
                                Layout<Shape<_4, _1, _1>>{},
                                Tile<_64, _64, _8>{});
  //copy share o to register using unversial copy uint128
  auto s2g_copyO_atom = Copy_Atom<UniversalCopy<uint128_t>, scalar_t>{};
  auto s2g_copyO = make_tiled_copy(s2g_copyO_atom, 
                                  Layout<Shape<_16,_8>, Stride<_8, _1>>{},
                                  Layout<Shape<_1, _8>>{});
  size_t smem_elems_half = (bN + bK + bK) * head_dim;
  size_t smem_elems_float = bN * head_dim ; // sQ, sK, sV, sO, sP
  size_t smem_half_bytes = smem_elems_half * sizeof(scalar_t);
  size_t smem_bytes = ((smem_half_bytes + alignof(float) - 1) / alignof(float)) * alignof(float)
                    + smem_elems_float * sizeof(float);


  using GLayoutQ_t  = std::decay_t<decltype(gQ)>;
  using GLayoutK_t  = std::decay_t<decltype(gK)>;
  using GLayoutV_t  = std::decay_t<decltype(gV)>;
  using GLayoutO_t  = std::decay_t<decltype(gO)>;
  using SLayoutQ_t  = std::decay_t<decltype(sQ)>;
  using SLayoutK_t  = std::decay_t<decltype(sK)>;
  using SLayoutV_t  = std::decay_t<decltype(sV)>;
  using SLayoutVTrans_t = std::decay_t<decltype(sV_trans)>;
  using SLayoutO_t  = std::decay_t<decltype(sO)>;
  using TileG2SQ_t  = std::decay_t<decltype(g2s_copyQ)>;
  using TileG2SK_t  = std::decay_t<decltype(g2s_copyK)>;
  using TileG2SV_t  = std::decay_t<decltype(g2s_copyV)>;
  using AtomS2RQ_t = std::decay_t<decltype(s2r_copyQ_atom)>;
  using AtomS2RK_t = std::decay_t<decltype(s2r_copyK_atom)>;
  using AtomS2RV_t = std::decay_t<decltype(s2r_copyV_atom)>;
  using MMA1_t      = std::decay_t<decltype(mma1)>;
  using MMA2_t      = std::decay_t<decltype(mma2)>;
  using TiledS2GO_t = std::decay_t<decltype(s2g_copyO)>;

  dim3 block(128);
  // assum q_len % 16 == 0
  assert(q_len % 64 == 0);
  dim3 grid(batch_size * num_heads * q_len / bN);

  using Kernel_t = void (*)(
      const scalar_t*, const scalar_t*, const scalar_t*,
      scalar_t*, float, bool,
      GLayoutQ_t, GLayoutK_t, GLayoutV_t, GLayoutO_t,
      SLayoutQ_t, SLayoutK_t, SLayoutV_t, SLayoutO_t, SLayoutVTrans_t,
      TileG2SQ_t, TileG2SK_t, TileG2SV_t,
      AtomS2RQ_t, AtomS2RK_t, AtomS2RV_t,
      MMA1_t, MMA2_t, 
      TiledS2GO_t
  );

  Kernel_t kernel_ptr =
      flash_attention_kernel<
          GLayoutQ_t, GLayoutK_t, GLayoutV_t, GLayoutO_t,
          SLayoutQ_t, SLayoutK_t, SLayoutV_t, SLayoutO_t, SLayoutVTrans_t,
          TileG2SQ_t, TileG2SK_t, TileG2SV_t,
          AtomS2RQ_t, AtomS2RK_t, AtomS2RV_t,
          MMA1_t, MMA2_t, 
          TiledS2GO_t
      >;

  CudaCheck(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(kernel_ptr),
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_bytes
  ), "cudaFuncSetAttribute for flash attention");

  kernel_ptr<<<grid, block, smem_bytes, stream>>>(q.ptr, k.ptr, v.ptr, out->ptr, dropout, causal,
                                                  gQ, gK, gV, gO,
                                                  sQ, sK, sV, sO, sV_trans,
                                                  g2s_copyQ, g2s_copyK, g2s_copyV,
                                                  s2r_copyQ_atom, s2r_copyK_atom, s2r_copyV_atom,
                                                  mma1, mma2,
                                                  s2g_copyO);
  CudaCheck(cudaGetLastError(), "launch flash attention kernel");


  return;
}

void FlashAttentionForward(const CudaArray& q, const CudaArray& k, const CudaArray& v,
                          CudaArray* out, size_t batch_size, size_t num_heads,
                          size_t q_len, size_t kv_len,
                          size_t head_dim, float dropout, bool causal) {
  LaunchFlashAttentionForward(q, k, v, out, batch_size, num_heads, q_len, kv_len,
                              head_dim, dropout, causal, 0);
}

float FlashAttentionForwardBenchmark(const CudaArray& q, const CudaArray& k, const CudaArray& v,
                                     CudaArray* out, size_t batch_size, size_t num_heads,
                                     size_t q_len, size_t kv_len,
                                     size_t head_dim, float dropout, bool causal,
                                     int repeats) {
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
                                  head_dim, dropout, causal, stream);
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
  m.def("flash_attention_forward_benchmark", FlashAttentionForwardBenchmark);
#else
  m.attr("__has_flash_attention_stub__") = false;
#endif
}
