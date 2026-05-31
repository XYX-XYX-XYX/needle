#include "flash_attention_common.h"

#include <cfloat>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cute/util/print.hpp>
#include <cute/util/print_latex.hpp>

namespace needle {
namespace cuda {

using namespace cute;
typedef cute::half_t scalar_t;

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
template <int Arch, typename GLayoutQ, typename GLayoutK, typename GLayoutV, typename GLayoutO,
          typename SLayoutQ, typename SLayoutK, typename SLayoutV, typename SLayoutO, typename SLayoutVTrans_t,
          typename TileG2SQ, typename TileG2SK, typename TileG2SV,
          typename AtomS2RQ, typename AtomS2RK, typename AtomS2RV,
          typename MMA1, typename MMA2, 
          typename TiledS2GO>
__global__ void flash_attention_kernel_arch(const scalar_t* q, const scalar_t* k, const scalar_t* v,
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
  scalar_t* oShmem = reinterpret_cast<scalar_t*>(smem + smem_offset);
  
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
  auto gK = local_tile(K(batch_id, head_id, _, _),
                       make_tile(size<0>(sKL), size<1>(sKL)),
                       make_coord(_, _0{})); //(bK, head_dim, k)
  auto gV = local_tile(V(batch_id, head_id, _, _),
                        make_tile(size<0>(sVL), size<1>(sVL)),
                        make_coord(_, _0{})); //(bK, head_dim, k)

  auto gO = local_tile(O(batch_id, head_id, _, _),
                       make_tile(size<0>(sOL), size<1>(sOL)),
                       make_coord(seq_id, _0{}));               // (bN, head_dim)

  Tensor sQ = make_tensor(make_smem_ptr<scalar_t>(qShmem), sQL);   // (bN, head_dim)
  Tensor sK = make_tensor(make_smem_ptr<scalar_t>(kShmem), sKL);   // (bK, head_dim, kstage)
  Tensor sV = make_tensor(make_smem_ptr<scalar_t>(vShmem), sVL);   // (bK, head_dim, kstage)
  Tensor sV_trans  = make_tensor(make_smem_ptr<scalar_t>(vShmem), sVTransL);   // (head_dim, bK, kstage)
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
  auto tKgK = g2s_thr_copyK.partition_S(gK); //(CPY, CPY_M, CPY_N, k)
  auto tKsK = g2s_thr_copyK.partition_D(sK);   // (CPY, CPY_M, CPY_N, kstage)

  // load gV to sV
  auto g2s_thr_copyV = g2s_copyV.get_slice(threadIdx.x);
  auto tVgV = g2s_thr_copyV.partition_S(gV); //(CPY, CPY_M, CPY_N, k)
  auto tVsV = g2s_thr_copyV.partition_D(sV);   // (CPY, CPY_M, CPY_N, kstage)

  // mma1 and alloc register
  auto thr_mma1 = mma1.get_slice(threadIdx.x);
  auto mma1sQ = thr_mma1.partition_A(sQ);   // (MMA, Mma_M, Mma_K)
  auto mma1sK = thr_mma1.partition_B(sK(_, _, 0));    // (MMA, Mma_N, Mma_K)
  auto mma1sP = thr_mma1.partition_C(sP);   // (MMA, Mma_M, Mma_N)
  auto mma1rQ = thr_mma1.make_fragment_A(mma1sQ);   // (MMA, Mma_M, Mma_K)
  auto mma1rK = thr_mma1.make_fragment_B(mma1sK);   // (MMA, Mma_N, Mma_K)
  auto mma1rP = thr_mma1.make_fragment_C(mma1sP);   // (MMA, Mma_M, Mma_N)
  //clear(mma1rP);

  // mma2 and alloc register
  auto thr_mma2 = mma2.get_slice(threadIdx.x);
  auto mma2sP = thr_mma2.partition_A(sP);   // (MMA, Mma_M, Mma_K)
  auto mma2sV = thr_mma2.partition_B(sV_trans(_, _, 0)); //(MMA, Mma_N, Mma_K)
  auto mma2sO = thr_mma2.partition_C(sO);   // (MMA, Mma_M, Mma_N)
  // auto mma2rP = thr_mma2.make_fragment_A(mma2sP);   // (MMA, Mma_M, Mma_K)
  // auto mma2rP_float = make_fragment_like<float>(mma2sP);   // (MMA, Mma_M, Mma_K)
  auto mma2rP = make_fragment_like<scalar_t>(mma2sP);
  auto mma2rV = thr_mma2.make_fragment_B(mma2sV); //(MMA, Mma_N, Mma_K)
  auto mma2rO = thr_mma2.make_fragment_C(mma2sO);
  auto mma2gO = thr_mma2.partition_C(gO);

  auto mma1rP_mma2_view = make_tensor(mma1rP.data(), mma2rP.layout());

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
  auto tKsK_s2r = s2r_thr_copyK.partition_S(sK); //(CPY, CPY_M, CPY_N, kstage)
  auto tKrK_s2r = s2r_thr_copyK.retile_D(mma1rK);//(CPY, CPY_M, CPY_N)
  //load sV to rV
  auto s2r_tiled_copyV = make_tiled_copy_B(s2r_copyV_atom, mma2);
  auto s2r_thr_copyV = s2r_tiled_copyV.get_slice(threadIdx.x);
  auto tVsV_s2r = s2r_thr_copyV.partition_S(sV_trans);//(CPY, CPY_M, CPY_N, kstage)
  auto tVrV_s2r = s2r_thr_copyV.retile_D(mma2rV);//(CPY, CPY_M, CPY_N)

  //load sO to gO
  auto g2s_thr_copyO = s2g_copyO.get_slice(threadIdx.x);
  auto tOsO = g2s_thr_copyO.partition_S(sO);
  auto tOgO = g2s_thr_copyO.partition_D(gO);

  if(thread0()) {
    // print(tKgK); print("\n");
    // print(tKsK); print("\n");
    // print(tKsK_s2r); print("\n");
    // print(tKrK_s2r); print("\n");
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
    // print(mma2gO); print("\n");
    // print(mma2sO); print("\n");
    // print(mma2rO); print("\n");
  }

  // load gQ to sQ and load sQ to register
  copy(g2s_copyQ, tQgQ, tQsQ);
  cp_async_fence();

  // load gK_0 to sK gV_0 to sV
  // auto gK = local_tile(K(batch_id, head_id, _, _),
  //                      make_tile(size<0>(sKL), size<1>(sKL)),
  //                      make_coord(0, 0)); // (bK, head_dim)
  // auto tKgK = g2s_thr_copyK.partition_S(gK);
  int istage = 0;
  int kstage = 0;
  copy(g2s_copyK, tKgK(_, _, _, 0), tKsK(_, _, _, istage));  
  cp_async_fence();
  copy(g2s_copyV, tVgV(_, _, _, 0), tVsV(_, _, _, istage));
  cp_async_fence();
  istage++;
  cp_async_wait<2>();
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

    // auto gV = local_tile(V(batch_id, head_id, _, _),
    //                    make_tile(size<0>(sVL), size<1>(sVL)),
    //                    make_coord(i, 0));                    // (bK, head_dim)
    // auto tVgV = g2s_thr_copyV.partition_S(gV);
    // copy(g2s_copyV, tVgV, tVsV);
    // cp_async_fence();
  
    //load gV_{i+1) to sV and load sK_{i+1} to sK
    if(i != size<2>(gKL) / size<0>(sKL) - 1) {
      copy(g2s_copyK, tKgK(_, _, _, i + 1), tKsK(_, _, _, istage));
      cp_async_fence();
      copy(g2s_copyV, tVgV(_, _, _, i + 1), tVsV(_, _, _, istage));
      cp_async_fence();
      istage = (istage + 1) % 2;
      cp_async_wait<3>();
    } else {
      cp_async_wait<1>();
    }
    __syncthreads();

    // cp_async_wait<1>();
    // __syncthreads();
    // load sK to register
    copy(s2r_tiled_copyK, tKsK_s2r(_, _, _, kstage), tKrK_s2r);
    // __syncthreads();//防止sK被下一轮的global to shared 覆盖

    // //load gV_{i+1} to sV
    // if(i != size<2>(gKL) / size<0>(sKL) - 1) {
    //   gK = local_tile(K(batch_id, head_id, _, _),
    //                      make_tile(size<0>(sKL), size<1>(sKL)),
    //                      make_coord(i + 1, 0));                    // (bK, head_dim)
    //   tKgK = g2s_thr_copyK.partition_S(gK);
    //   copy(g2s_copyK, tKgK, tKsK);
    //   cp_async_fence();
    // }

    gemm(mma1, mma1rQ, mma1rK, mma1rP);


    if(i == 0) {
      online_softmax<true>(mma1rP, row_max, row_sum, mma2rO);
    } else {
      online_softmax<false>(mma1rP, row_max, row_sum, mma2rO);
    }
    
    convert_type_out(mma1rP_mma2_view, mma2rP);
    
    if(i != size<2>(gKL) / size<0>(sKL) - 1) {
      cp_async_wait<2>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    //copy sV to register 
    copy(s2r_tiled_copyV, tVsV_s2r(_, _, _, kstage), tVrV_s2r);
    kstage = (kstage + 1) % 2;

    gemm(mma2, mma2rP, mma2rV, mma2rO);
    //copy(mma2rO, mma2sO);
    __syncthreads();
  }

  float row_sum_inv[2];
  row_sum_inv[0] = 1.0f / row_sum[0];
  row_sum_inv[1] = 1.0f / row_sum[1];
  for(size_t i = 0; i < size<2>(mma2rO); i++) {
    for(size_t j = 0; j < size<0, 0>(mma2rO); j++) {
      mma2rO(make_coord(j, 0), 0, i) = mma2rO(make_coord(j, 0), 0, i) * row_sum_inv[0];
      mma2rO(make_coord(j, 1), 0, i) = mma2rO(make_coord(j, 1), 0, i) * row_sum_inv[1];
    }
  }
  
  copy(mma2rO, mma2sO);
  // copy(mma2rO, mma2gO);
  __syncthreads();

  copy(s2g_copyO, tOsO, tOgO);
  return;
}

template <int Arch>
void LaunchFlashAttentionForwardForArch(const CudaArray& q, const CudaArray& k, const CudaArray& v,
                                        CudaArray* out, size_t batch_size, size_t num_heads,
                                        size_t q_len, size_t kv_len,
                                        size_t head_dim, float dropout, bool causal,
                                        const char* kernel_name, cudaStream_t stream) {
  (void)sizeof(cutlass::Status);

  const size_t q_size = batch_size * num_heads * q_len * head_dim;
  const size_t kv_size = batch_size * num_heads * kv_len * head_dim;
  const size_t out_size = q_size;

  if (q.size != q_size) throw std::runtime_error("flash attention q tensor size does not match the provided metadata");
  if (k.size != kv_size) throw std::runtime_error("flash attention k tensor size does not match the provided metadata");
  if (v.size != kv_size) throw std::runtime_error("flash attention v tensor size does not match the provided metadata");
  if (out->size != out_size) throw std::runtime_error("flash attention output tensor size does not match the provided metadata");
  if (head_dim != 64) throw std::runtime_error("flash attention kernels currently support head_dim=64");

  std::ostringstream msg;
  msg << "flash attention " << kernel_name << " kernel"
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
  auto kstage = Int<2>{};

  //auto sQ = make_layout(make_shape(bN,  _64{}), LayoutRight{});
  auto sQ = composition(Swizzle<3,3,3>{}, make_layout(make_shape(bN,  _64{}), LayoutRight{}));
  //auto sK = make_layout(make_shape(bK,  _64{}), LayoutRight{});
  auto sK = composition(Swizzle<3,3,3>{}, make_layout(make_shape(bK,  _64{}, kstage), make_stride(_64{}, _1{}, _64{} * bK)));
  //auto sV = make_layout(make_shape(bK,  _64{}), LayoutRight{});
  auto sV = composition(Swizzle<3,3,3>{}, make_layout(make_shape(bK, _64{}, kstage), make_stride(_64{}, _1{}, _64{} * bK)));
  auto sV_trans = composition(Swizzle<3,3,3>{}, make_layout(make_shape(_64{}, bK, kstage), LayoutLeft{}));
  //auto sV_trans = make_layout(make_shape(_64{}, bK), LayoutLeft{});
  //auto sO = make_layout(make_shape(bN,  _64{}), LayoutRight{});
  auto sO = composition(Swizzle<3,3,3>{}, make_layout(make_shape(bN, _64{}), LayoutRight{}));

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
  auto s2r_copyQ_atom = Copy_Atom<SM75_U32x4_LDSM_N, scalar_t>{};
  auto s2r_copyK_atom = Copy_Atom<SM75_U32x4_LDSM_N, scalar_t>{};
  auto s2r_copyV_atom = Copy_Atom<SM75_U16x8_LDSM_T, scalar_t>{};
  //mma1
  auto mma1 = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
                                Layout<Shape<_4, _1, _1>>{},
                                Tile<_64, _64, _16>{});
  auto mma2 = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
                                Layout<Shape<_4, _1, _1>>{},
                                Tile<_64, _64, _16>{});
  //copy share o to register using unversial copy uint128
  auto s2g_copyO_atom = Copy_Atom<UniversalCopy<uint128_t>, scalar_t>{};
  auto s2g_copyO = make_tiled_copy(s2g_copyO_atom, 
                                  Layout<Shape<_16,_8>, Stride<_8, _1>>{},
                                  Layout<Shape<_1, _8>>{});
  size_t smem_elems_half = (bN + bK * 2 + bK * 2 + bN) * head_dim;
  size_t smem_half_bytes = smem_elems_half * sizeof(scalar_t);
  size_t smem_bytes = smem_half_bytes;


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
      flash_attention_kernel_arch<
          Arch, GLayoutQ_t, GLayoutK_t, GLayoutV_t, GLayoutO_t,
          SLayoutQ_t, SLayoutK_t, SLayoutV_t, SLayoutO_t, SLayoutVTrans_t,
          TileG2SQ_t, TileG2SK_t, TileG2SV_t,
          AtomS2RQ_t, AtomS2RK_t, AtomS2RV_t,
          MMA1_t, MMA2_t, 
          TiledS2GO_t
      >;

  std::string attr_context = std::string("cudaFuncSetAttribute for flash attention ") + kernel_name;
  CudaCheck(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(kernel_ptr),
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_bytes
  ), attr_context.c_str());

  kernel_ptr<<<grid, block, smem_bytes, stream>>>(reinterpret_cast<const scalar_t*>(q.ptr),
                                                  reinterpret_cast<const scalar_t*>(k.ptr),
                                                  reinterpret_cast<const scalar_t*>(v.ptr),
                                                  reinterpret_cast<scalar_t*>(out->ptr), dropout, causal,
                                                  gQ, gK, gV, gO,
                                                  sQ, sK, sV, sO, sV_trans,
                                                  g2s_copyQ, g2s_copyK, g2s_copyV,
                                                  s2r_copyQ_atom, s2r_copyK_atom, s2r_copyV_atom,
                                                  mma1, mma2,
                                                  s2g_copyO);
  std::string launch_context = std::string("launch flash attention ") + kernel_name + " kernel";
  CudaCheck(cudaGetLastError(), launch_context.c_str());


  return;
}


void LaunchFlashAttentionForwardSm80(const CudaArray& q, const CudaArray& k, const CudaArray& v,
                  CudaArray* out, size_t batch_size, size_t num_heads,
                  size_t q_len, size_t kv_len,
                  size_t head_dim, float dropout, bool causal,
                  cudaStream_t stream) {
  LaunchFlashAttentionForwardForArch<80>(q, k, v, out, batch_size, num_heads, q_len, kv_len,
                                             head_dim, dropout, causal, "sm80", stream);
}

}  // namespace cuda
}  // namespace needle
