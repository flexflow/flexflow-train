#include "op-attrs/parallel_tensor_dim_degrees.h"
#include <doctest/doctest.h>
#include "test/utils/doctest/fmt/unordered_map.h"
#include "test/utils/doctest/fmt/unordered_set.h"

using namespace ::FlexFlow; 

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_parallel_tensor_degree_map") {
    ParallelTensorDimDegrees degrees = ParallelTensorDimDegrees{
      SumDegree{3},
      DiscardCopyDegree{1},
      FFOrdered<int>{
        1,
        2,
        1,
      },
    };

    std::unordered_map<parallel_tensor_dim_idx_t, int> result = get_parallel_tensor_degree_map(degrees);
    std::unordered_map<parallel_tensor_dim_idx_t, int> correct = {
      {parallel_tensor_dim_idx_t{ReplicaType::SUM}, 3},
      {parallel_tensor_dim_idx_t{ReplicaType::DISCARD_COPY}, 1},
      {parallel_tensor_dim_idx_t{ff_dim_t{0}}, 1},
      {parallel_tensor_dim_idx_t{ff_dim_t{1}}, 2},
      {parallel_tensor_dim_idx_t{ff_dim_t{2}}, 1},
    };

    CHECK(result == correct);
  }

  TEST_CASE("get_parallel_tensor_space_coordinates") {
    ParallelTensorDimDegrees degrees = ParallelTensorDimDegrees{
      SumDegree{3},
      DiscardCopyDegree{1},
      FFOrdered<int>{
        1,
        2,
        1,
      },
    };

    std::unordered_set<ParallelTensorSpaceCoordinate> result = get_parallel_tensor_space_coordinates(degrees);
    std::unordered_set<ParallelTensorSpaceCoordinate> correct = {
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/0,
        /*discard_copy_idx=*/0,
        /*shard_idxs=*/FFOrdered<int>{0, 0, 0},
      },
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/1,
        /*discard_copy_idx=*/0,
        /*shard_idxs=*/FFOrdered<int>{0, 0, 0},
      },
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/2,
        /*discard_copy_idx=*/0,
        /*shard_idxs=*/FFOrdered<int>{0, 0, 0},
      },
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/0,
        /*discard_copy_idx=*/0,
        /*shard_idxs=*/FFOrdered<int>{0, 1, 0},
      },
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/1,
        /*discard_copy_idx=*/0,
        /*shard_idxs=*/FFOrdered<int>{0, 1, 0},
      },
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/2,
        /*discard_copy_idx=*/0,
        /*shard_idxs=*/FFOrdered<int>{0, 1, 0},
      },
    };

    CHECK(result == correct);
  }
}
