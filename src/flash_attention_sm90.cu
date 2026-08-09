#include "flash_attention_common.h"
#include "flash_attention_tile_scheduler.h"

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
#include <cutlass/arch/barrier.h>




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
CUTLASS_DEVICE void online_softmax(FragmentP& rP, float dscale, float* row_max, float* row_sum, float* scale) 
{
  float row_max_new[2] = {row_max[0], row_max[1]};

  //scale rP by head dim scale 
  #pragma unroll
  for(size_t i = 0; i < size(rP); i++) {
    rP(i) = rP(i) * dscale;
  }

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

  // float scale[2];
  scale[0] = __expf(row_max[0] - row_max_new[0]);
  scale[1] = __expf(row_max[1] - row_max_new[1]);

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

template <int Arch, typename TileScheduler, typename GLayoutO, typename Prod_Shape,
          typename TmaQ, typename TmaK, typename TmaV, typename TmaO,
          typename SLayoutQ, typename SLayoutK, typename SLayoutV, typename SLayoutO,
          typename MMA1, typename MMA2,
          typename AtomR2SO>
__global__ __launch_bounds__(384, 1) 
void flash_attention_kernel_arch(const scalar_t* q, const scalar_t* k, const scalar_t* v,
                                            scalar_t* o, float dropout, bool causal, Prod_Shape prod_shape,
                                            GLayoutO gOL,
                                            CUTLASS_GRID_CONSTANT const TmaQ tmaQ, CUTLASS_GRID_CONSTANT const TmaK tmaK, CUTLASS_GRID_CONSTANT const TmaV tmaV, CUTLASS_GRID_CONSTANT const TmaO tmaO,
                                            SLayoutQ sQL, SLayoutK sKL, SLayoutV sVL, SLayoutO sOL,
                                            MMA1 mma1, MMA2 mma2,
                                            AtomR2SO r2s_copyO_atom,
                                            typename TileScheduler::Params tile_scheduler_params)
{

  TileScheduler tile_scheduler(tile_scheduler_params);
  // (void)tile_scheduler;

  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int lane_predicate = cute::elect_one_sync();
  const int warp_group_idx = cutlass::canonical_warp_group_idx();
  const int warp_group_thread_idx = threadIdx.x % 128;

  //prefech for tensormap
  if(warp_idx == 0 && lane_predicate) {
    cute::prefetch_tma_descriptor(tmaQ.get_tma_descriptor());
    cute::prefetch_tma_descriptor(tmaK.get_tma_descriptor());
    cute::prefetch_tma_descriptor(tmaV.get_tma_descriptor());
    cute::prefetch_tma_descriptor(tmaO.get_tma_descriptor());
  }
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

  //pipeline for k v
  using TmaPipeline = typename cutlass::PipelineTmaAsync<_2{}>;
  using GenericPipeline = typename cutlass::PipelineAsync<_2{}>;
  using PipelineState = typename cutlass::PipelineState<_2{}>;

  //pipeline for q
  using TmaPipeline_Q = typename cutlass::PipelineTmaAsync<_1{}>;
  using PipelineState_Q = typename cutlass::PipelineState<_1{}>;

  TmaPipeline::Params tma_pipeline_params;
  tma_pipeline_params.transaction_bytes = sizeof(scalar_t) * size<0>(sKL) * size<1>(sKL);
  tma_pipeline_params.role = warp_group_idx == 0 ?  TmaPipeline::ThreadCategory::Producer : TmaPipeline::ThreadCategory::Consumer;
  tma_pipeline_params.is_leader = (lane_predicate && warp_idx == 0);
  tma_pipeline_params.num_consumers = 256;

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
  
  TmaPipeline_Q::Params tma_pipeline_params_q;
  tma_pipeline_params_q.transaction_bytes = sizeof(scalar_t) * size(sQL);
  tma_pipeline_params_q.role = warp_group_idx == 0 ?  TmaPipeline_Q::ThreadCategory::Producer : TmaPipeline_Q::ThreadCategory::Consumer;
  tma_pipeline_params_q.is_leader = (lane_predicate && warp_idx == 0);
  tma_pipeline_params_q.num_consumers = 256;

  //pipeline for Q
  auto *tma_pipeline_q_storage = reinterpret_cast<typename TmaPipeline_Q::SharedStorage *>(smem + smem_offset);
  smem_offset += sizeof(typename TmaPipeline_Q::SharedStorage);
  TmaPipeline_Q pipeline_q(*tma_pipeline_q_storage, tma_pipeline_params_q, Shape<_1, _1, _1>{});
  PipelineState_Q tma_q_write_state = cutlass::make_producer_start_state<TmaPipeline_Q>();
  PipelineState_Q tma_q_read_state;
  // //barrier for q
  // constexpr int tma_transaction_q_bytes = sizeof(scalar_t) * size(sQL);
  // using BarType = cutlass::arch::ClusterTransactionBarrier;
  // uint64_t* tma_barrier_q = reinterpret_cast<uint64_t*>(smem + smem_offset);
  // if(lane_predicate && warp_group_idx == 0 && warp_idx == 0) {
  //   BarType::init(tma_barrier_q, 1);
  // }
  __syncthreads();

  if(thread0()) {
    // print("mma1 accmulator thread share memory view:"); print(mma1sP); print("\n");
    // print("mma1 accmulator thread register view:"); print(mma1rP); print("\n");
    // print("mma2 operand A thread share memory view:"); print(mma2sP); print("\n");
    // print("mma2 operand A thread register view:"); print(mma2rP); print("\n");
    // print("mma1:\n"); print_latex(mma1); print("\n");
    // print("mma2:\n"); print_latex(mma2); print("\n");
  }

  if(warp_group_idx == 0) {
    //producer 
    cutlass::arch::warpgroup_reg_dealloc<24>();

    auto mQ = tmaQ.get_tma_tensor(prod_shape);
    auto mK = tmaK.get_tma_tensor(prod_shape);
    auto mV = tmaV.get_tma_tensor(prod_shape);
    auto sQ = make_tensor(make_smem_ptr<scalar_t>(qShmem), sQL);   // (bN, head_dim, Nwarps)
    auto sK = make_tensor(make_smem_ptr<scalar_t>(kShmem), sKL);   // (bK, head_dim, kstage)
    auto sV = make_tensor(make_smem_ptr<scalar_t>(vShmem), sVL);   // (bK, head_dim, kstage)

    for(auto work_tile = tile_scheduler.get_initial_work(); work_tile.is_valid(); work_tile = tile_scheduler.get_next_work(work_tile)) {
      int batch_id = work_tile.batch_id();
      int head_id = work_tile.head_id();
      int seq_id = work_tile.query_tile_id();
      
      auto gQ_cta = local_tile(mQ(batch_id, head_id, _, _),
                          make_tile(size<0>(sQL) * 2, size<1>(sQL)),
                          make_coord(seq_id, _0{}));               // (bN * 2, head_dim)
      auto gQ_div = flat_divide(gQ_cta, make_tile(size<0>(sQL), size<1>(sQL)));   // (bN, head_dim, Nwarps, 1)
      auto gQ = gQ_div(_, _, _, _0{});   // (bN, head_dim, Nwarps)

      auto gK = local_tile(mK(batch_id, head_id, _, _),
                          make_tile(size<0>(sKL), size<1>(sKL)),
                          make_coord(_, _0{})); //(bK, head_dim, k)

      auto gV = local_tile(mV(batch_id, head_id, _, _),
                          make_tile(size<0>(sVL), size<1>(sVL)),
                          make_coord(_, _0{})); //(bK, head_dim, k)

      // load global to shared
      auto [tQgQ, tQsQ] = tma_partition(tmaQ, Int<0>{}, Layout<_1>{},
                                        group_modes<0, 2>(sQ), group_modes<0, 2>(gQ));//(TMA, Nwarps)
      auto [tKgK, tKsK] = tma_partition(tmaK, Int<0>{}, Layout<_1>{},
                                        group_modes<0, 2>(sK), group_modes<0, 2>(gK));//(TMA, kstage)
      auto [tVgV, tVsV] = tma_partition(tmaV, Int<0>{}, Layout<_1>{},
                                        group_modes<0, 2>(sV), group_modes<0, 2>(gV));//(TMA, kstage)
      if(lane_predicate && warp_idx == 0) {
        // global q to share q
        pipeline_q.producer_acquire(tma_q_write_state);
        auto tma_barrier_q = pipeline_q.producer_get_barrier(tma_q_write_state);
        copy(tmaQ.with(*tma_barrier_q), tQgQ, tQsQ);
        ++tma_q_write_state;
      }
      
      int num_kv_tiles = get<2>(prod_shape) / size<0>(sKL);
      for(size_t i = 0; i < num_kv_tiles; i++) {
        if(lane_predicate && warp_idx == 0) {
          pipeline_k.producer_acquire(tma_k_write_state);
          auto tma_barrier_k = pipeline_k.producer_get_barrier(tma_k_write_state);
          copy(tmaK.with(*tma_barrier_k), tKgK(_, i), tKsK(_, tma_k_write_state.index()));
          ++tma_k_write_state;

          pipeline_v.producer_acquire(tma_v_write_state);
          auto tma_barrier_v = pipeline_v.producer_get_barrier(tma_v_write_state);
          copy(tmaV.with(*tma_barrier_v), tVgV(_, i), tVsV(_, tma_v_write_state.index()));
          ++tma_v_write_state;
        }
      }
    }

  } else {
    //consumer 
    cutlass::arch::warpgroup_reg_alloc<160>();
    const int consumer_idx = warp_group_idx - 1;
    constexpr uint32_t kConsumer1Issued = 0;
    constexpr uint32_t kConsumer2Issued = 1;
    constexpr int kConsumerThreads = 2 * cutlass::NumThreadsPerWarpGroup; // 256
    auto pingpong_wait = [&](bool first_gemm_group) {
      if (consumer_idx == 0) {
        // Consumer 1 第一轮直接启动；后续等待 Consumer 2
        if (!first_gemm_group) {
          cutlass::arch::NamedBarrier::sync(
              kConsumerThreads, kConsumer2Issued);
        }
      } else {
        // Consumer 2 每轮都等待 Consumer 1
        cutlass::arch::NamedBarrier::sync(
            kConsumerThreads, kConsumer1Issued);
      }
    };
    auto pingpong_arrive = [&]() {
      if (consumer_idx == 0) {
        cutlass::arch::NamedBarrier::arrive(
            kConsumerThreads, kConsumer1Issued);
      } else {
        cutlass::arch::NamedBarrier::arrive(
            kConsumerThreads, kConsumer2Issued);
      }
    };
    bool first_gemm_group = true;
    
    auto sQ_all = make_tensor(make_smem_ptr<scalar_t>(qShmem), sQL);   // (bN, head_dim, Nwarps)
    auto sQ = sQ_all(_, _, consumer_idx);   // (bN, head_dim)
    auto sK = make_tensor(make_smem_ptr<scalar_t>(kShmem), sKL);   // (bK, head_dim, kstage)
    auto sV = make_tensor(make_smem_ptr<scalar_t>(vShmem), sVL);   // (bK, head_dim, kstage)
    auto sVt_layout = composition(sVL, make_layout(make_shape(size<1>(sVL), size<0>(sVL), size<2>(sVL)), make_stride(size<0>(sVL), _1{}, size<0>(sVL) * size<1>(sVL))));
    auto sVt = make_tensor(make_smem_ptr<scalar_t>(vShmem), sVt_layout);   // (head_dim, bK, kstage)
    auto sO_all = make_tensor(make_smem_ptr<scalar_t>(oShmem), sOL);   // (bN, head_dim, Nwarps)
    auto sO = sO_all(_, _, consumer_idx);   // (bN, head_dim)
    auto sP = make_coord_tensor(make_layout(make_shape(size<0>(sQL), size<0>(sKL))));  // (bN, bK)
    // mma1
    auto thr_mma1 = mma1.get_slice(warp_group_thread_idx);
    auto mma1sQ = thr_mma1.partition_A(sQ);   // (MMA, Mma_M, Mma_K)
    auto mma1sK = thr_mma1.partition_B(sK);    // (MMA, Mma_N, Mma_K, kstage)
    auto mma1sP = thr_mma1.partition_C(sP);   // (MMA, Mma_M, Mma_N)
    auto mma1rQ = thr_mma1.make_fragment_A(mma1sQ);   // (MMA, Mma_M, Mma_K)
    auto mma1rK = thr_mma1.make_fragment_B(mma1sK);   // (MMA, Mma_N, Mma_K, kstage)
    auto mma1rP = thr_mma1.make_fragment_C(mma1sP);   // (MMA, Mma_M, Mma_N)
    // mma2 
    auto thr_mma2 = mma2.get_slice(warp_group_thread_idx);
    auto mma2sP = thr_mma2.partition_A(sP);   // (MMA, Mma_M, Mma_K)
    auto mma2sV = thr_mma2.partition_B(sVt); // (MMA, Mma_N, Mma_K, kstage)
    auto mma2sO = thr_mma2.partition_C(sO);   // (MMA, Mma_M, Mma_N)
    auto mma2rP = thr_mma2.make_fragment_A(mma2sP);   // (MMA, Mma_M, Mma_K)
    auto mma2rV = thr_mma2.make_fragment_B(mma2sV); //(MMA, Mma_N, Mma_K, kstage)
    auto mma2rO = thr_mma2.make_fragment_C(mma2sO); // (MMA, Mma_M, Mma_N)
    auto mma2rO_half = make_fragment_like<scalar_t>(mma2rO);
    auto mma1rP_mma2_view = make_tensor(mma1rP.data(), mma2rP.layout()); //(MMA, Mma_M, Mma_K)
    // rO to sO
    auto r2s_tiled_copyO = make_tiled_copy_C(r2s_copyO_atom, mma2);
    auto r2s_thr_copyO = r2s_tiled_copyO.get_slice(warp_group_thread_idx);
    auto tOrO_r2s = r2s_thr_copyO.retile_S(mma2rO_half);
    auto tOsO_r2s = r2s_thr_copyO.partition_D(sO);
    //tma store gO 
    auto mO = tmaO.get_tma_tensor(prod_shape);

    float d_scale = rsqrtf(float(get<3>(prod_shape)));
    for(auto work_tile = tile_scheduler.get_initial_work(); work_tile.is_valid(); work_tile = tile_scheduler.get_next_work(work_tile)) {
      int batch_id = work_tile.batch_id();
      int head_id = work_tile.head_id();
      int seq_id = work_tile.query_tile_id();
      auto gO_cta = local_tile(mO(batch_id, head_id, _, _),
                        make_tile(size<0>(sOL) * 2, size<1>(sOL)),
                        make_coord(seq_id, _0{}));               // (bN * 2, head_dim)
      auto gO_div = flat_divide(gO_cta, make_tile(size<0>(sOL), size<1>(sOL)));   // (bN, head_dim, Nwarps, 1)
      auto gO = gO_div(_, _, consumer_idx, _0{});   // (bN, head_dim)
      auto [tOgO, tOsO] = tma_partition(tmaO, int{0}, Layout<_1>{},
                                      group_modes<0, 2>(sO), group_modes<0, 2>(gO));//(TMA)

      //process
      float row_max[2] = {-FLT_MAX, -FLT_MAX};
      float row_sum[2] = {0.0f, 0.0f};
      float scale[2];
      clear(mma2rO);
      //wait for tma for Q to finish
      pipeline_q.consumer_wait(tma_q_read_state);

      //wait for tma for k_0 to finsh
      pipeline_k.consumer_wait(tma_k_read_state);
      //gemm 1 for q and k_0
      pingpong_wait(first_gemm_group);
      clear(mma1rP);
      warpgroup_fence_operand(mma1rP);
      warpgroup_arrive();
      gemm(mma1, mma1rQ, mma1rK(_, _, _, tma_k_read_state.index()), mma1rP);
      warpgroup_commit_batch();
      pingpong_arrive();
      first_gemm_group = false;
      //wait for gemm1_0 to finish and release k_0
      warpgroup_wait<0>();
      warpgroup_fence_operand(mma1rP);
      pipeline_k.consumer_release(tma_k_read_state);
      ++tma_k_read_state;
      online_softmax<true>(mma1rP, d_scale, row_max, row_sum, scale);
      //mma1rP to mma2rP
      convert_type_out(mma1rP_mma2_view, mma2rP);
          int num_kv_tiles = get<2>(prod_shape) / size<0>(sKL);
      for(size_t i = 0; i < num_kv_tiles - 1; i++) {
        //wait tma k_i+1 to finish and launch gemm1_i+1
        pipeline_k.consumer_wait(tma_k_read_state);
        pingpong_wait(first_gemm_group);
        clear(mma1rP);
        warpgroup_fence_operand(mma1rP);
        warpgroup_arrive();
        gemm(mma1, mma1rQ, mma1rK(_, _, _, tma_k_read_state.index()), mma1rP);
        warpgroup_commit_batch();

        //wait tma v_i to finish and launch gemm2_i which consumes mma2rP_i and produces mma2rO_i
        pipeline_v.consumer_wait(tma_v_read_state);
        warpgroup_fence_operand(mma2rO);
        warpgroup_fence_operand(mma2rP);
        warpgroup_arrive();
        gemm(mma2, mma2rP, mma2rV(_, _, _, tma_v_read_state.index()), mma2rO);
        warpgroup_commit_batch();
        pingpong_arrive();

        //wait gemm1_i+1 to finish and release k_i+1
        warpgroup_wait<1>();
        warpgroup_fence_operand(mma1rP);
        pipeline_k.consumer_release(tma_k_read_state);
        ++tma_k_read_state;

        //online softmax for mma1rP_i+1
        online_softmax<false>(mma1rP, d_scale, row_max, row_sum, scale);
        
        //wait gemm2_i to finish and release v_i
        warpgroup_wait<0>();
        warpgroup_fence_operand(mma2rO);
        warpgroup_fence_operand(mma2rP);
        pipeline_v.consumer_release(tma_v_read_state);
        ++tma_v_read_state;

        //scale rO_i and prepare for the next gemm2
        #pragma unroll
        for(size_t j = 0; j < size<0, 2>(mma2rO); j++) {
          for(size_t k = 0; k < size<0, 0>(mma2rO); k++) {
            mma2rO(make_coord(k, 0, j), 0, 0) = mma2rO(make_coord(k, 0, j), 0, 0) * scale[0];
            mma2rO(make_coord(k, 1, j), 0, 0) = mma2rO(make_coord(k, 1, j), 0, 0) * scale[1];
          }
        }
        //mma1rP_i+1 to mma2rP_i+1
        convert_type_out(mma1rP_mma2_view, mma2rP);
      }

      //num_tiles - 1, only do gemm2
      {
        pipeline_v.consumer_wait(tma_v_read_state);
        pingpong_wait(first_gemm_group);
        warpgroup_fence_operand(mma2rO);
        warpgroup_fence_operand(mma2rP);
        warpgroup_arrive();
        gemm(mma2, mma2rP, mma2rV(_, _, _, tma_v_read_state.index()), mma2rO);
        warpgroup_commit_batch();       
        pingpong_arrive();
        warpgroup_wait<0>();
        warpgroup_fence_operand(mma2rO);
        warpgroup_fence_operand(mma2rP);
        pipeline_q.consumer_release(tma_q_read_state);
        ++tma_q_read_state;
        pipeline_v.consumer_release(tma_v_read_state);
        ++tma_v_read_state;
      }

      //epilogue
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
      cute::tma_store_wait<0>();
      //copy O from register to share memory
      copy(r2s_copyO_atom, tOrO_r2s, tOsO_r2s);

      tma_store_fence();
      //all mma threads arrive and proceed 
      cutlass::arch::NamedBarrier::sync(kConsumerThreads,
                                              cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      
      
      // tma copy O share memory  to global memory
      if(warp_group_thread_idx == 0) {
        copy(tmaO, tOsO, tOgO);
      }
      cute::tma_store_arrive();
    }

  }
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
  auto bK = Int<128>{};
  auto kstage = Int<2>{};
  auto NWarps = Int<2>{};


  auto sQ = tile_to_shape(GMMA::Layout_K_SW128_Atom<scalar_t>{}, make_shape(bN, _64{}, NWarps));
  auto sK = tile_to_shape(GMMA::Layout_K_SW128_Atom<scalar_t>{}, make_shape(bK, _64{}, kstage));
  auto sV = tile_to_shape(GMMA::Layout_K_SW128_Atom<scalar_t>{}, make_shape(bK, _64{}, kstage));
  auto sO = tile_to_shape(GMMA::Layout_K_SW128_Atom<scalar_t>{}, make_shape(bN, _64{}, NWarps));

  auto mQ = make_tensor(reinterpret_cast<const scalar_t*>(q.ptr), gQ);
  auto mK = make_tensor(reinterpret_cast<const scalar_t*>(k.ptr), gK);
  auto mV = make_tensor(reinterpret_cast<const scalar_t*>(v.ptr), gV);
  auto mO = make_tensor(reinterpret_cast<scalar_t*>(out->ptr), gO);

  Copy_Atom tmaQ = make_tma_atom(SM90_TMA_LOAD{}, mQ, sQ(_, _, 0), make_shape(_1{}, _1{}, bN, _64{}));
  Copy_Atom tmaK = make_tma_atom(SM90_TMA_LOAD{}, mK, sK(_, _, 0), make_shape(_1{}, _1{}, bK, _64{}));
  Copy_Atom tmaV = make_tma_atom(SM90_TMA_LOAD{}, mV, sV(_, _, 0), make_shape(_1{}, _1{}, bK, _64{}));
  Copy_Atom tmaO = make_tma_atom(SM90_TMA_STORE{}, mO, sO(_, _, 0), make_shape(_1{}, _1{}, bN, _64{}));


  auto mma1 = make_tiled_mma(SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
  auto mma2 = make_tiled_mma(SM90_64x64x16_F32F16F16_RS<GMMA::Major::K, GMMA::Major::MN>{});
  

  auto r2s_copyO_atom = Copy_Atom<SM90_U32x4_STSM_N, scalar_t>{};


constexpr size_t kGmmaSmemAlignment = 128;

size_t smem_bytes = 0;
smem_bytes = align_up_bytes(smem_bytes, kGmmaSmemAlignment);
smem_bytes += (bN * NWarps + bK * kstage + bK * kstage + bN * NWarps) * head_dim  * sizeof(scalar_t); // sQ, sK, sV, sO
smem_bytes += 2 * sizeof(typename cutlass::PipelineTmaAsync<_2{}>::SharedStorage) + 
                sizeof(typename cutlass::PipelineAsync<_2{}>::SharedStorage) + 
                sizeof(typename cutlass::PipelineTmaAsync<_1{}>::SharedStorage); // pipeline for k, v, q


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
  using TileScheduler_t = LinearTileScheduler;
  using TileSchedulerParams_t = typename TileScheduler_t::Params;

  TileSchedulerParams_t tile_scheduler_params{
      static_cast<int>(batch_size),
      static_cast<int>(num_heads),
      static_cast<int>(q_len / (bN * 2))};

  dim3 block(384);
  // assum q_len % 64 == 0
  assert(q_len % 128 == 0);
  assert(kv_len == q_len);
  dim3 grid(132);

  using Kernel_t = void (*)(
      const scalar_t*, const scalar_t*, const scalar_t*,
      scalar_t*, float, bool, Prod_Shape_t,
      GLayoutO_t,
      TmaQ_t,  TmaK_t, TmaV_t, TmaO_t,
      SLayoutQ_t, SLayoutK_t, SLayoutV_t, SLayoutO_t,
      MMA1_t, MMA2_t, 
      AtomR2SO_t,
      TileSchedulerParams_t
  );

  Kernel_t kernel_ptr =
      flash_attention_kernel_arch<
        Arch, TileScheduler_t, GLayoutO_t, Prod_Shape_t,
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
                                                  r2s_copyO_atom,
                                                  tile_scheduler_params);
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
