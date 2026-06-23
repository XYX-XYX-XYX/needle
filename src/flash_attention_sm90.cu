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

#include "cutlass/device_kernel.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/pipeline.hpp"
#include <cutlass/arch/reg_reconfig.h>




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


template<bool isFirst, typename FragmentP>
CUTLASS_DEVICE void online_softmax(FragmentP& rP, float* row_max, float* row_sum, float* scale) 
{
  float row_max_new[2] = {row_max[0], row_max[1]};

  //get local thread row max
  for(size_t i = 0; i < size<0, 2>(rP); i++) {
    for(size_t j = 0; j < size<0, 0>(rP); j++) {
      row_max_new[0] = rP(make_coord(j, 0, i), 0, 0) > row_max_new[0] ? rP(make_coord(j, 0, i), 0, 0) : row_max_new[0];
      row_max_new[1] = rP(make_coord(j, 1, i), 0, 0) > row_max_new[1] ? rP(make_coord(j, 1, i), 0, 0) : row_max_new[1];
    }
  }
  //shuffle  every thread get the max value of the row
  unsigned mask = unsigned(-1);
  row_max_new[0] = fmaxf(row_max_new[0], __shfl_xor_sync(mask, row_max_new[0], 1, 4));
  row_max_new[0] = fmaxf(row_max_new[0], __shfl_xor_sync(mask, row_max_new[0], 2, 4));
  row_max_new[1] = fmaxf(row_max_new[1], __shfl_xor_sync(mask, row_max_new[1], 1, 4));
  row_max_new[1] = fmaxf(row_max_new[1], __shfl_xor_sync(mask, row_max_new[1], 2, 4));


  //get local thread sum
  float row_sum_new[2] = {0.0f, 0.0f};
  for(size_t i = 0; i < size<0, 2>(rP); i++) {
    for(size_t j = 0; j < size<0, 0>(rP); j++) {
      rP(make_coord(j, 0, i), 0, 0) = __expf(rP(make_coord(j, 0, i), 0, 0) - row_max_new[0]);
      rP(make_coord(j, 1, i), 0, 0) = __expf(rP(make_coord(j, 1, i), 0, 0) - row_max_new[1]);
      row_sum_new[0] += rP(make_coord(j, 0, i), 0, 0);
      row_sum_new[1] += rP(make_coord(j, 1, i), 0, 0);

    }
  }
  //shuffle to get the sum of the row
  row_sum_new[0] += __shfl_xor_sync(mask, row_sum_new[0], 1, 4);
  row_sum_new[0] += __shfl_xor_sync(mask, row_sum_new[0], 2, 4);
  row_sum_new[1] += __shfl_xor_sync(mask, row_sum_new[1], 1, 4);
  row_sum_new[1] += __shfl_xor_sync(mask, row_sum_new[1], 2, 4);

  // float alph[2];
  scale[0] = __expf(row_max[0] - row_max_new[0]);
  scale[1] = __expf(row_max[1] - row_max_new[1]);
  //update rO
  // if(!isFirst) {
  //   for(size_t i = 0; i < size<0, 2>(rO); i++) {
  //     for(size_t j = 0; j < size<0, 0>(rO); j++) {
  //       rO(make_coord(j, 0, i), 0, 0) = rO(make_coord(j, 0, i), 0, 0) * alph[0];
  //       rO(make_coord(j, 1, i), 0, 0) = rO(make_coord(j, 1, i), 0, 0) * alph[1];
  //     }
  //   }
  // }

  //update row_max and row_sum
  if(!isFirst) {
    row_sum[0] = row_sum_new[0] + row_sum[0] * scale[0];
    row_sum[1] = row_sum_new[1] + row_sum[1] * scale[1];
  } else {
    row_sum[0] = row_sum_new[0];
    row_sum[1] = row_sum_new[1];
  }

  row_max[0] = row_max_new[0];
  row_max[1] = row_max_new[1];

  return;

}

CUTLASS_HOST_DEVICE
constexpr size_t align_up_bytes(size_t offset, size_t alignment) {
  // alignment 必须是 2 的幂
  return (offset + alignment - 1) & ~(alignment - 1);
}

template <int Arch, typename GLayoutO, typename Prod_Shape,
          typename TmaQ, typename TmaK, typename TmaV, typename TmaO,
          typename SLayoutQ, typename SLayoutK, typename SLayoutV, typename SLayoutO,
          typename MMA1, typename MMA2,
          typename AtomR2SO>
__global__ void flash_attention_kernel_arch(const scalar_t* q, const scalar_t* k, const scalar_t* v,
                                            scalar_t* o, float dropout, bool causal, Prod_Shape prod_shape,
                                            GLayoutO gOL,
                                            CUTLASS_GRID_CONSTANT const TmaQ tmaQ, CUTLASS_GRID_CONSTANT const TmaK tmaK, CUTLASS_GRID_CONSTANT const TmaV tmaV, CUTLASS_GRID_CONSTANT const TmaO tmaO,
                                            SLayoutQ sQL, SLayoutK sKL, SLayoutV sVL, SLayoutO sOL,
                                            MMA1 mma1, MMA2 mma2,
                                            AtomR2SO r2s_copyO_atom)
{

  // 保证整个动态 shared memory 的起始地址至少 128B 对齐
  extern __shared__ __align__(128) unsigned char smem[];
  size_t smem_offset = 0;
  scalar_t* qShmem =reinterpret_cast<scalar_t*>(smem + smem_offset);
  smem_offset += static_cast<size_t>(cosize(sQL)) * sizeof(scalar_t);
  scalar_t* kShmem = reinterpret_cast<scalar_t*>(smem + smem_offset);
  smem_offset += static_cast<size_t>(cosize(sKL)) * sizeof(scalar_t);
  scalar_t* vShmem = reinterpret_cast<scalar_t*>(smem + smem_offset);
  smem_offset += static_cast<size_t>(cosize(sVL)) * sizeof(scalar_t);
  scalar_t* oShmem = reinterpret_cast<scalar_t*>(smem + smem_offset);
  smem_offset += static_cast<size_t>(cosize(sOL)) * sizeof(scalar_t);
  // uint64_t* tma_barrier_q = reinterpret_cast<uint64_t*>(smem + smem_offset);
  // smem_offset += sizeof(uint64_t);
  // uint64_t* tma_barrier_k = reinterpret_cast<uint64_t*>(smem + smem_offset);
  // smem_offset += sizeof(uint64_t);
  // uint64_t* tma_barrier_v = reinterpret_cast<uint64_t*>(smem + smem_offset);
  // smem_offset += sizeof(uint64_t);
  // uint64_t* barrier_p = reinterpret_cast<uint64_t*>(smem + smem_offset);

  
  float row_max[2] = {-FLT_MAX, -FLT_MAX};
  float row_sum[2] = {0.0f, 0.0f};
  float scale[2];

  int bid = blockIdx.x;
  int num_tiles = get<2>(prod_shape) / size<0>(sQL);
  int seq_id = bid % num_tiles;
  int head_id = (bid / num_tiles) % get<1>(prod_shape);
  int batch_id = bid / (num_tiles * get<1>(prod_shape));

  auto mQ = tmaQ.get_tma_tensor(prod_shape);
  auto mK = tmaK.get_tma_tensor(prod_shape);
  auto mV = tmaV.get_tma_tensor(prod_shape);
  auto mO = tmaO.get_tma_tensor(prod_shape);

  auto gQ = local_tile(mQ(batch_id, head_id, _, _),
                       make_tile(size<0>(sQL), size<1>(sQL)),
                       make_coord(seq_id, _0{}));               // (bN, head_dim)

  auto gK = local_tile(mK(batch_id, head_id, _, _),
                       make_tile(size<0>(sKL), size<1>(sKL)),
                       make_coord(_, _0{})); //(bK, head_dim, k)
  auto gV = local_tile(mV(batch_id, head_id, _, _),
                       make_tile(size<0>(sVL), size<1>(sVL)),
                       make_coord(_, _0{})); //(bK, head_dim, k)
  auto gO = local_tile(mO(batch_id, head_id, _, _),
                       make_tile(size<0>(sOL), size<1>(sOL)),
                       make_coord(seq_id, _0{}));               // (bN, head_dim)
  auto sQ = make_tensor(make_smem_ptr<scalar_t>(qShmem), sQL);   // (bN, head_dim)
  auto sK = make_tensor(make_smem_ptr<scalar_t>(kShmem), sKL);   // (bK, head_dim, kstage)
  auto sV = make_tensor(make_smem_ptr<scalar_t>(vShmem), sVL);   // (bK, head_dim, kstage)
  auto sVt_layout = composition(sVL, make_layout(make_shape(size<1>(sVL), size<0>(sVL), size<2>(sVL)), make_stride(size<0>(sVL), _1{}, size<0>(sVL) * size<1>(sVL))));
  auto sVt = make_tensor(make_smem_ptr<scalar_t>(vShmem), sVt_layout);   // (head_dim, bK, kstage)
  auto sO = make_tensor(make_smem_ptr<scalar_t>(oShmem), sOL);   // (bN, head_dim)
  auto sP = make_coord_tensor(make_layout(make_shape(size<0>(sQL), size<0>(sKL))));  // (bN, bK)


  // load global to shared
  auto [tQgQ, tQsQ] = tma_partition(tmaQ, Int<0>{}, Layout<_1>{},
                                    group_modes<0, 2>(sQ), group_modes<0, 2>(gQ));//(TMA)
  auto [tKgK, tKsK] = tma_partition(tmaK, Int<0>{}, Layout<_1>{},
                                    group_modes<0, 2>(sK), group_modes<0, 2>(gK));//(TMA, kstage)
  auto [tVgV, tVsV] = tma_partition(tmaV, Int<0>{}, Layout<_1>{},
                                    group_modes<0, 2>(sV), group_modes<0, 2>(gV));//(TMA, kstage)
  auto [tOgO, tOsO] = tma_partition(tmaO, int{0}, Layout<_1>{},
                                    group_modes<0, 2>(sO), group_modes<0, 2>(gO));//(TMA)
  
  // mma1
  auto thr_mma1 = mma1.get_slice(threadIdx.x);
  auto mma1sQ = thr_mma1.partition_A(sQ);   // (MMA, Mma_M, Mma_K)
  auto mma1sK = thr_mma1.partition_B(sK);    // (MMA, Mma_N, Mma_K, kstage)
  auto mma1sP = thr_mma1.partition_C(sP);   // (MMA, Mma_M, Mma_N)
  auto mma1rQ = thr_mma1.make_fragment_A(mma1sQ);   // (MMA, Mma_M, Mma_K)
  auto mma1rK = thr_mma1.make_fragment_B(mma1sK);   // (MMA, Mma_N, Mma_K, kstage)
  auto mma1rP = thr_mma1.make_fragment_C(mma1sP);   // (MMA, Mma_M, Mma_N)
  // auto mma1rP_stage2 = thr_mma1.make_fragment_C(mma1sP);   // (MMA, Mma_M, Mma_N)
  // clear(mma1rP);

  // mma2 
  auto thr_mma2 = mma2.get_slice(threadIdx.x);
  auto mma2sP = thr_mma2.partition_A(sP);   // (MMA, Mma_M, Mma_K)
  auto mma2sV = thr_mma2.partition_B(sVt); // (MMA, Mma_N, Mma_K, kstage)
  auto mma2sO = thr_mma2.partition_C(sO);   // (MMA, Mma_M, Mma_N)
  auto mma2rP = thr_mma2.make_fragment_A(mma2sP);   // (MMA, Mma_M, Mma_K)
  auto mma2rV = thr_mma2.make_fragment_B(mma2sV); //(MMA, Mma_N, Mma_K, kstage)
  auto mma2rO = thr_mma2.make_fragment_C(mma2sO); // (MMA, Mma_M, Mma_N)
  auto mma2rO_half = make_fragment_like<scalar_t>(mma2rO);
  clear(mma2rO);

  auto mma1rP_mma2_view = make_tensor(mma1rP.data(), mma2rP.layout()); //(MMA, Mma_M, Mma_K)
  // auto mma1rP_stage2_mma2_view = make_tensor(mma1rP_stage2.data(), mma2rP.layout()); //(MMA, Mma_M, Mma_K)

  //sO to gO
  // auto g2s_thr_copyO = s2g_copyO.get_slice(threadIdx.x);
  // auto tOsO = g2s_thr_copyO.partition_S(sO);
  // auto tOgO = g2s_thr_copyO.partition_D(gO);
  // rO to sO
  auto r2s_tiled_copyO = make_tiled_copy_C(r2s_copyO_atom, mma2);
  auto r2s_thr_copyO = r2s_tiled_copyO.get_slice(threadIdx.x);
  auto tOrO_r2s = r2s_thr_copyO.retile_S(mma2rO_half);
  auto tOsO_r2s = r2s_thr_copyO.partition_D(sO);

  //pipeline
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  using TmaPipeline = typename cutlass::PipelineTmaAsync<_2{}>;
  using GenericPipeline = typename cutlass::PipelineAsync<_2{}>;
  using PipelineState = typename cutlass::PipelineState<_2{}>;

  TmaPipeline::Params tma_pipeline_params;
  tma_pipeline_params.transaction_bytes = sizeof(scalar_t) * size<0>(sK) * size<1>(sK);
  tma_pipeline_params.role = TmaPipeline::ThreadCategory::ProducerConsumer;
  tma_pipeline_params.is_leader = (lane_predicate && warp_idx == 0);
  tma_pipeline_params.num_consumers = blockDim.x;

  GenericPipeline::Params generic_pipeline_params;
  generic_pipeline_params.role = GenericPipeline::ThreadCategory::ProducerConsumer;
  generic_pipeline_params.producer_arv_count = blockDim.x;
  generic_pipeline_params.consumer_arv_count = blockDim.x;
  generic_pipeline_params.dst_blockid = 0;

  //pipeline for K
  auto *tma_pipeline_k_storage = reinterpret_cast<typename TmaPipeline::SharedStorage *>(smem + smem_offset);
  smem_offset += sizeof(typename TmaPipeline::SharedStorage);
  TmaPipeline pipeline_k(*tma_pipeline_k_storage, tma_pipeline_params, Shape<_1, _1, _1>{});
  PipelineState tma_k_write_state = cutlass::make_producer_start_state<TmaPipeline>();
  PipelineState tma_k_read_state;

  //pipeline for V
  auto *tma_pipeline_v_storage = reinterpret_cast<typename TmaPipeline::SharedStorage *>(smem + smem_offset);
  smem_offset += sizeof(typename TmaPipeline::SharedStorage);
  TmaPipeline pipeline_v(*tma_pipeline_v_storage, tma_pipeline_params, Shape<_1, _1, _1>{});
  PipelineState tma_v_write_state = cutlass::make_producer_start_state<TmaPipeline>();
  PipelineState tma_v_read_state;

  //pipeline for P
  // auto *pipeline_p_storage = reinterpret_cast<typename GenericPipeline::SharedStorage *>(smem + smem_offset);
  // smem_offset += sizeof(typename GenericPipeline::SharedStorage);
  // GenericPipeline pipeline_p(*pipeline_p_storage, generic_pipeline_params);
  // PipelineState tma_p_write_state = cutlass::make_producer_start_state<GenericPipeline>();
  // PipelineState tma_p_read_state;

  //barrier for q
  using BarType = cutlass::arch::ClusterTransactionBarrier;
  uint64_t* tma_barrier_q = reinterpret_cast<uint64_t*>(smem + smem_offset);
  if(lane_predicate && warp_idx == 0){
    BarType::init(tma_barrier_q, 1);
  }
  // cutlass::arch::warpgroup_reg_alloc<256>();
  __syncthreads();

  if(thread0()) {
    // print("mma1 accmulator thread share memory view:"); print(mma1sP); print("\n");
    // print("mma1 accmulator thread register view:"); print(mma1rP); print("\n");
    // print("mma2 operand A thread share memory view:"); print(mma2sP); print("\n");
    // print("mma2 operand A thread register view:"); print(mma2rP); print("\n");
    // print("mma1:\n"); print_latex(mma1); print("\n");
    // print("mma2:\n"); print_latex(mma2); print("\n");
  }
  constexpr int tma_transaction_q_bytes = sizeof(scalar_t) * size(sQ);

  if(lane_predicate && warp_idx == 0) {
    // global q to share q
    BarType::arrive_and_expect_tx(tma_barrier_q, tma_transaction_q_bytes);
    copy(tmaQ.with(*tma_barrier_q), tQgQ, tQsQ);
    // global k_0 to share k_0
    pipeline_k.producer_acquire(tma_k_write_state);
    auto tma_barrier_k = pipeline_k.producer_get_barrier(tma_k_write_state);
    copy(tmaK.with(*tma_barrier_k), tKgK(_, 0), tKsK(_, tma_k_write_state.index()));
    // pipeline_k.producer_commit(tma_k_write_state);
    ++tma_k_write_state;
    //global k_1 to share k_1
    pipeline_k.producer_acquire(tma_k_write_state);
    auto tma_barrier_k_1 = pipeline_k.producer_get_barrier(tma_k_write_state);
    copy(tmaK.with(*tma_barrier_k_1), tKgK(_, 1), tKsK(_, tma_k_write_state.index()));
    ++tma_k_write_state;
    //global v_0 to share v_0
    pipeline_v.producer_acquire(tma_v_write_state);
    auto tma_barrier_v = pipeline_v.producer_get_barrier(tma_v_write_state);
    copy(tmaV.with(*tma_barrier_v), tVgV(_, 0), tVsV(_, tma_v_write_state.index()));
    ++tma_v_write_state;
  }
  //wait for tma for Q to finish
  BarType::wait(tma_barrier_q, 0);
  //wait for tma for k_0 to finsh
  pipeline_k.consumer_wait(tma_k_read_state);
  // pipeline_p.producer_acquire(tma_p_write_state);
  //gemm 1 for q and k_0
  clear(mma1rP);
  warpgroup_fence_operand(mma1rP);
  warpgroup_arrive();
  gemm(mma1, mma1rQ, mma1rK(_, _, _, tma_k_read_state.index()), mma1rP);
  warpgroup_commit_batch();

  //wait for gemm1_0 to finish which consume k_0 and produce p_0
  warpgroup_wait<0>();
  warpgroup_fence_operand(mma1rP);
  pipeline_k.consumer_release(tma_k_read_state);
  ++tma_k_read_state;
  // pipeline_p.producer_commit(tma_p_write_state);
  // ++tma_p_write_state;

  float d_scale = rsqrtf(float(get<3>(prod_shape)));
  #pragma unroll
  for(size_t i = 0; i < size(mma1rP); i++) {
    mma1rP(i) = mma1rP(i) * d_scale;
  }
  online_softmax<true>(mma1rP, row_max, row_sum, scale);
  convert_type_out(mma1rP_mma2_view, mma2rP);

  for(size_t i = 0; i < num_tiles - 2; i++) {

    if(lane_predicate && warp_idx == 0) {
      //wait for consumer to release 
      pipeline_k.producer_acquire(tma_k_write_state);
      pipeline_v.producer_acquire(tma_v_write_state);
      //global to share for k_i+2 and v_i+1
      auto tma_barrier_k = pipeline_k.producer_get_barrier(tma_k_write_state);
      auto tma_barrier_v = pipeline_v.producer_get_barrier(tma_v_write_state);
      //copy global to share for k_i+2 and v_i+1
      copy(tmaK.with(*tma_barrier_k), tKgK(_, i + 2), tKsK(_, tma_k_write_state.index()));
      copy(tmaV.with(*tma_barrier_v), tVgV(_, i + 1), tVsV(_, tma_v_write_state.index()));
      ++tma_k_write_state;
      ++tma_v_write_state;
    }


    pipeline_k.consumer_wait(tma_k_read_state);
    clear(mma1rP);
    warpgroup_fence_operand(mma1rP);
    warpgroup_arrive();
    gemm(mma1, mma1rQ, mma1rK(_, _, _, tma_k_read_state.index()), mma1rP);
    warpgroup_commit_batch();

    pipeline_v.consumer_wait(tma_v_read_state);
    warpgroup_fence_operand(mma2rO);
    warpgroup_fence_operand(mma2rP);
    warpgroup_arrive();
    gemm(mma2, mma2rP, mma2rV(_, _, _, tma_v_read_state.index()), mma2rO);
    warpgroup_commit_batch();

    warpgroup_wait<1>();
    warpgroup_fence_operand(mma1rP);
    pipeline_k.consumer_release(tma_k_read_state);
    ++tma_k_read_state;



    #pragma unroll
    for(size_t j = 0; j < size(mma1rP); j++) {
      mma1rP(j) = mma1rP(j) * d_scale;
    }
    online_softmax<false>(mma1rP, row_max, row_sum, scale);
    
    warpgroup_wait<0>();
    warpgroup_fence_operand(mma2rO);
    warpgroup_fence_operand(mma2rP);
    pipeline_v.consumer_release(tma_v_read_state);
    ++tma_v_read_state;
    //scale rO
    #pragma unroll
    for(size_t j = 0; j < size<0, 2>(mma2rO); j++) {
      for(size_t k = 0; k < size<0, 0>(mma2rO); k++) {
        mma2rO(make_coord(k, 0, j), 0, 0) = mma2rO(make_coord(k, 0, j), 0, 0) * scale[0];
        mma2rO(make_coord(k, 1, j), 0, 0) = mma2rO(make_coord(k, 1, j), 0, 0) * scale[1];
      }
    }
    convert_type_out(mma1rP_mma2_view, mma2rP);
    
    __syncthreads();
  }

  {
    size_t tile_idx = num_tiles - 2;
    if(lane_predicate && warp_idx == 0) {
      //wait for consumer to release 
      pipeline_v.producer_acquire(tma_v_write_state);
      //global to share for k_i+2 and v_i+1
      auto tma_barrier_v = pipeline_v.producer_get_barrier(tma_v_write_state);
      //copy global to share for k_i+2 and v_i+1
      copy(tmaV.with(*tma_barrier_v), tVgV(_, tile_idx + 1), tVsV(_, tma_v_write_state.index()));
      ++tma_v_write_state;
    }


    pipeline_k.consumer_wait(tma_k_read_state);
    clear(mma1rP);
    warpgroup_fence_operand(mma1rP);
    warpgroup_arrive();
    gemm(mma1, mma1rQ, mma1rK(_, _, _, tma_k_read_state.index()), mma1rP);
    warpgroup_commit_batch();

    pipeline_v.consumer_wait(tma_v_read_state);
    warpgroup_fence_operand(mma2rO);
    warpgroup_fence_operand(mma2rP);
    warpgroup_arrive();
    gemm(mma2, mma2rP, mma2rV(_, _, _, tma_v_read_state.index()), mma2rO);
    warpgroup_commit_batch();

    warpgroup_wait<1>();
    warpgroup_fence_operand(mma1rP);
    pipeline_k.consumer_release(tma_k_read_state);
    ++tma_k_read_state;


    #pragma unroll
    for(size_t j = 0; j < size(mma1rP); j++) {
      mma1rP(j) = mma1rP(j) * d_scale;
    }
    online_softmax<false>(mma1rP, row_max, row_sum, scale);
    
    warpgroup_wait<0>();
    warpgroup_fence_operand(mma2rO);
    warpgroup_fence_operand(mma2rP);
    pipeline_v.consumer_release(tma_v_read_state);
    ++tma_v_read_state;
    //scale rO
    #pragma unroll
    for(size_t j = 0; j < size<0, 2>(mma2rO); j++) {
      for(size_t k = 0; k < size<0, 0>(mma2rO); k++) {
        mma2rO(make_coord(k, 0, j), 0, 0) = mma2rO(make_coord(k, 0, j), 0, 0) * scale[0];
        mma2rO(make_coord(k, 1, j), 0, 0) = mma2rO(make_coord(k, 1, j), 0, 0) * scale[1];
      }
    }
    convert_type_out(mma1rP_mma2_view, mma2rP);
    
    __syncthreads();
  }

  {

    pipeline_v.consumer_wait(tma_v_read_state);
    warpgroup_fence_operand(mma2rO);
    warpgroup_fence_operand(mma2rP);
    warpgroup_arrive();
    gemm(mma2, mma2rP, mma2rV(_, _, _, tma_v_read_state.index()), mma2rO);
    warpgroup_commit_batch();       
    warpgroup_wait<0>();
    warpgroup_fence_operand(mma2rO);
    warpgroup_fence_operand(mma2rP);
    
    __syncthreads();
  }

  //finalize scaling output by row_sum
  float row_sum_inv[2];
  row_sum_inv[0] = 1.0f / row_sum[0];
  row_sum_inv[1] = 1.0f / row_sum[1];
  for(size_t i = 0; i < size<0, 2>(mma2rO); i++) {
    for(size_t j = 0; j < size<0, 0>(mma2rO); j++) {
      mma2rO(make_coord(j, 0, i), 0, 0) = mma2rO(make_coord(j, 0, i), 0, 0) * row_sum_inv[0];
      mma2rO(make_coord(j, 1, i), 0, 0) = mma2rO(make_coord(j, 1, i), 0, 0) * row_sum_inv[1];
    }
  }

  //convert output to fp16
  convert_type_out(mma2rO, mma2rO_half);
  //copy o from register to share memory
  copy(r2s_copyO_atom, tOrO_r2s, tOsO_r2s);
  __syncthreads();
  tma_store_fence();
  // copy share memory output to global memory
  if(warp_idx == 0 && lane_predicate) {
    copy(tmaO, tOsO, tOgO);
    // tma_store_arrive();
  }
  // tma_store_wait<0>();
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
  auto prod_shape = make_shape(batch_size, num_heads, q_len, head_dim);

  auto bN = Int<64>{};
  auto bK = Int<64>{};
  auto kstage = Int<2>{};


  auto sQ = tile_to_shape(GMMA::Layout_K_SW128_Atom<scalar_t>{}, make_shape(bN, _64{}));
  auto sK = tile_to_shape(GMMA::Layout_K_SW128_Atom<scalar_t>{}, make_shape(bK, _64{}, kstage));
  auto sV = tile_to_shape(GMMA::Layout_K_SW128_Atom<scalar_t>{}, make_shape(bK, _64{}, kstage));
  // auto sO = composition(Swizzle<3,3,3>{}, make_layout(make_shape(bN, _64{}), LayoutRight{}));
  // auto sO_tma = composition(Swizzle<3,4,3>{}, make_layout(make_shape(bN, _64{}), LayoutRight{}));
  auto sO = tile_to_shape(GMMA::Layout_K_SW128_Atom<scalar_t>{}, make_shape(bN, _64{}));

  auto mQ = make_tensor(reinterpret_cast<const scalar_t*>(q.ptr), gQ);
  auto mK = make_tensor(reinterpret_cast<const scalar_t*>(k.ptr), gK);
  auto mV = make_tensor(reinterpret_cast<const scalar_t*>(v.ptr), gV);
  auto mO = make_tensor(reinterpret_cast<scalar_t*>(out->ptr), gO);

  Copy_Atom tmaQ = make_tma_atom(SM90_TMA_LOAD{}, mQ, sQ, make_shape(_1{}, _1{}, bN, _64{}));
  Copy_Atom tmaK = make_tma_atom(SM90_TMA_LOAD{}, mK, sK(_, _, 0), make_shape(_1{}, _1{}, bK, _64{}));
  Copy_Atom tmaV = make_tma_atom(SM90_TMA_LOAD{}, mV, sV(_, _, 0), make_shape(_1{}, _1{}, bK, _64{}));
  Copy_Atom tmaO = make_tma_atom(SM90_TMA_STORE{}, mO, sO, make_shape(_1{}, _1{}, bN, _64{}));


  auto mma1 = make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
  auto mma2 = make_tiled_mma(SM90_64x64x16_F32F16F16_RS<GMMA::Major::K, GMMA::Major::MN>{});
  

  auto r2s_copyO_atom = Copy_Atom<SM90_U32x4_STSM_N, scalar_t>{};


constexpr size_t kGmmaSmemAlignment = 128;

size_t smem_bytes = 0;
smem_bytes = align_up_bytes(smem_bytes, kGmmaSmemAlignment);
smem_bytes += (bN + bK * 2 + bK * 2 + bN) * head_dim * sizeof(scalar_t); // sQ, sK, sV, sO
smem_bytes += 2 * sizeof(typename cutlass::PipelineTmaAsync<_2{}>::SharedStorage) + 
                sizeof(typename cutlass::PipelineAsync<_2{}>::SharedStorage) +
                sizeof(uint64_t); // TMA barriers


  using Prod_Shape_t = decltype(prod_shape);
  using GLayoutO_t  = std::decay_t<decltype(gO)>;
  using TmaQ_t = std::decay_t<decltype(tmaQ)>;
  using TmaK_t = std::decay_t<decltype(tmaK)>;
  using TmaV_t = std::decay_t<decltype(tmaV)>;
  using TmaO_t = std::decay_t<decltype(tmaO)>;
  using SLayoutQ_t  = std::decay_t<decltype(sQ)>;
  using SLayoutK_t  = std::decay_t<decltype(sK)>;
  using SLayoutV_t  = std::decay_t<decltype(sV)>;
  using SLayoutO_t  = std::decay_t<decltype(sO)>;
  using MMA1_t      = std::decay_t<decltype(mma1)>;
  using MMA2_t      = std::decay_t<decltype(mma2)>;
  using AtomR2SO_t = std::decay_t<decltype(r2s_copyO_atom)>;

  dim3 block(128);
  // assum q_len % 64 == 0
  assert(q_len % 64 == 0);
  assert(kv_len == q_len);
  dim3 grid(batch_size * num_heads * q_len / bN);

  using Kernel_t = void (*)(
      const scalar_t*, const scalar_t*, const scalar_t*,
      scalar_t*, float, bool, Prod_Shape_t,
      GLayoutO_t,
      TmaQ_t,  TmaK_t, TmaV_t, TmaO_t,
      SLayoutQ_t, SLayoutK_t, SLayoutV_t, SLayoutO_t,
      MMA1_t, MMA2_t, 
      AtomR2SO_t
  );

  Kernel_t kernel_ptr =
      flash_attention_kernel_arch<
        Arch, GLayoutO_t, Prod_Shape_t,
        TmaQ_t, TmaK_t, TmaV_t, TmaO_t,
        SLayoutQ_t, SLayoutK_t, SLayoutV_t, SLayoutO_t,
        MMA1_t, MMA2_t,
        AtomR2SO_t
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
                                                  reinterpret_cast<scalar_t*>(out->ptr), dropout, causal, prod_shape,
                                                  gO,
                                                  tmaQ, tmaK, tmaV, tmaO,
                                                  sQ, sK, sV, sO,
                                                  mma1, mma2,
                                                  r2s_copyO_atom);
  std::string launch_context = std::string("launch flash attention ") + kernel_name + " kernel";
  CudaCheck(cudaGetLastError(), launch_context.c_str());


  return;
}


void LaunchFlashAttentionForwardSm90(const CudaArray& q, const CudaArray& k, const CudaArray& v,
                  CudaArray* out, size_t batch_size, size_t num_heads,
                  size_t q_len, size_t kv_len,
                  size_t head_dim, float dropout, bool causal,
                  cudaStream_t stream) {
  LaunchFlashAttentionForwardForArch<90>(q, k, v, out, batch_size, num_heads, q_len, kv_len,
                                             head_dim, dropout, causal, "sm90", stream);
}

}  // namespace cuda
}  // namespace needle
