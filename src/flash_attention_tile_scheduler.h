#pragma once

#include <cutlass/cutlass.h>

namespace needle {
namespace cuda {

// Assigns attention tiles to persistent CTAs in a deterministic, strided order:
//   CTA i -> i, i + gridDim.x, i + 2 * gridDim.x, ...
class LinearTileScheduler {
 public:
  struct Params {
    int num_batches;
    int num_heads;
    int num_query_tiles;

    CUTLASS_HOST_DEVICE
    int num_work_tiles() const {
      return num_batches * num_heads * num_query_tiles;
    }

    CUTLASS_HOST_DEVICE
    bool is_valid() const {
      return num_batches > 0 && num_heads > 0 && num_query_tiles > 0;
    }
  };

  class WorkTile {
   public:
    CUTLASS_HOST_DEVICE
    WorkTile()
        : linear_idx_(-1), num_work_tiles_(0), batch_id_(-1),
          head_id_(-1), query_tile_id_(-1) {}

    CUTLASS_HOST_DEVICE
    WorkTile(int linear_idx, int num_work_tiles, int batch_id, int head_id,
             int query_tile_id)
        : linear_idx_(linear_idx), num_work_tiles_(num_work_tiles),
          batch_id_(batch_id), head_id_(head_id),
          query_tile_id_(query_tile_id) {}

    CUTLASS_HOST_DEVICE
    bool is_valid() const {
      return linear_idx_ >= 0 && linear_idx_ < num_work_tiles_;
    }

    CUTLASS_HOST_DEVICE
    int linear_idx() const {
      return linear_idx_;
    }

    CUTLASS_HOST_DEVICE
    int batch_id() const {
      return batch_id_;
    }

    CUTLASS_HOST_DEVICE
    int head_id() const {
      return head_id_;
    }

    CUTLASS_HOST_DEVICE
    int query_tile_id() const {
      return query_tile_id_;
    }

   private:
    int linear_idx_;
    int num_work_tiles_;
    int batch_id_;
    int head_id_;
    int query_tile_id_;
  };

  CUTLASS_DEVICE
  explicit LinearTileScheduler(Params const& params) : params_(params) {}

  CUTLASS_DEVICE
  WorkTile get_initial_work() const {
    return make_work_tile(static_cast<int>(blockIdx.x));
  }

  CUTLASS_DEVICE
  WorkTile get_next_work(WorkTile const& current_work) const {
    return make_work_tile(current_work.linear_idx() +
                          static_cast<int>(gridDim.x));
  }

 private:
  CUTLASS_DEVICE
  WorkTile make_work_tile(int linear_idx) const {
    if (!params_.is_valid()) {
      return WorkTile{};
    }

    int block_decode = linear_idx;
    int query_tile_id = block_decode % params_.num_query_tiles;
    block_decode /= params_.num_query_tiles;
    int head_id = block_decode % params_.num_heads;
    int batch_id = block_decode / params_.num_heads;

    return WorkTile(linear_idx, params_.num_work_tiles(), batch_id, head_id,
                    query_tile_id);
  }

  Params params_;
};

}  // namespace cuda
}  // namespace needle
