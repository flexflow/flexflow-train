#include "op-attrs/parallel_tensor_dim_degrees.h"
#include <doctest/doctest.h>
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/set.h"

using namespace ::FlexFlow; 

static parallel_tensor_dim_idx_t shard_dim_idx_from_raw(int idx) {
  return parallel_tensor_dim_idx_t{ff_dim_t{nonnegative_int{idx}}};
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_parallel_tensor_degree_map") {
    ParallelTensorDimDegrees degrees = ParallelTensorDimDegrees{
      SumDegree{3_p},
      DiscardCopyDegree{1_p},
      FFOrdered{
        1_p,
        2_p,
        1_p,
      },
    };

    std::unordered_map<parallel_tensor_dim_idx_t, positive_int> result = get_parallel_tensor_degree_map(degrees);
    std::unordered_map<parallel_tensor_dim_idx_t, positive_int> correct = {
      {parallel_tensor_dim_idx_t{ReplicaType::SUM}, 3_p},
      {parallel_tensor_dim_idx_t{ReplicaType::DISCARD_COPY}, 1_p},
      {shard_dim_idx_from_raw(0), 1_p},
      {shard_dim_idx_from_raw(1), 2_p},
      {shard_dim_idx_from_raw(2), 1_p},
    };

    CHECK(result == correct);
  }

  TEST_CASE("get_parallel_tensor_space_coordinates") {
    ParallelTensorDimDegrees degrees = ParallelTensorDimDegrees{
      SumDegree{3_p},
      DiscardCopyDegree{1_p},
      FFOrdered{
        1_p,
        2_p,
        1_p,
      },
    };

    std::unordered_set<ParallelTensorSpaceCoordinate> result = get_parallel_tensor_space_coordinates(degrees);
    std::unordered_set<ParallelTensorSpaceCoordinate> correct = {
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/0_n,
        /*discard_copy_idx=*/0_n,
        /*shard_idxs=*/FFOrdered{0_n, 0_n, 0_n},
      },
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/1_n,
        /*discard_copy_idx=*/0_n,
        /*shard_idxs=*/FFOrdered{0_n, 0_n, 0_n},
      },
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/2_n,
        /*discard_copy_idx=*/0_n,
        /*shard_idxs=*/FFOrdered{0_n, 0_n, 0_n},
      },
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/0_n,
        /*discard_copy_idx=*/0_n,
        /*shard_idxs=*/FFOrdered{0_n, 1_n, 0_n},
      },
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/1_n,
        /*discard_copy_idx=*/0_n,
        /*shard_idxs=*/FFOrdered{0_n, 1_n, 0_n},
      },
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/2_n,
        /*discard_copy_idx=*/0_n,
        /*shard_idxs=*/FFOrdered{0_n, 1_n, 0_n},
      },
    };

    CHECK(result == correct);
  }

  TEST_CASE("get_nontrivial_parallel_tensor_dim_indices(ParallelTensorDimDegrees)") {
    SUBCASE("a replica dim has degree 1") {
      ParallelTensorDimDegrees degrees = ParallelTensorDimDegrees{
        SumDegree{3_p},
        DiscardCopyDegree{1_p},
        FFOrdered{4_p, 2_p, 4_p},
      };

      std::set<parallel_tensor_dim_idx_t> result = get_nontrivial_parallel_tensor_dim_indices(degrees);
      std::set<parallel_tensor_dim_idx_t> correct = {
        parallel_tensor_dim_idx_t{ReplicaType::SUM},
        shard_dim_idx_from_raw(0),
        shard_dim_idx_from_raw(1),
        shard_dim_idx_from_raw(2),
      };

      CHECK(result == correct);
    }

    SUBCASE("a shard dim has degree 1") {
      ParallelTensorDimDegrees degrees = ParallelTensorDimDegrees{
        SumDegree{3_p},
        DiscardCopyDegree{2_p},
        FFOrdered{1_p, 4_p, 1_p},
      };

      std::set<parallel_tensor_dim_idx_t> result = get_nontrivial_parallel_tensor_dim_indices(degrees);
      std::set<parallel_tensor_dim_idx_t> correct = {
        parallel_tensor_dim_idx_t{ReplicaType::SUM},
        parallel_tensor_dim_idx_t{ReplicaType::DISCARD_COPY},
        shard_dim_idx_from_raw(1),
      };

      CHECK(result == correct);
    }

    SUBCASE("no dims have degree 1") {
      ParallelTensorDimDegrees degrees = ParallelTensorDimDegrees{
        SumDegree{3_p},
        DiscardCopyDegree{2_p},
        FFOrdered{4_p, 2_p, 5_p},
      };

      std::set<parallel_tensor_dim_idx_t> result = get_nontrivial_parallel_tensor_dim_indices(degrees);
      std::set<parallel_tensor_dim_idx_t> correct = {
        parallel_tensor_dim_idx_t{ReplicaType::SUM},
        parallel_tensor_dim_idx_t{ReplicaType::DISCARD_COPY},
        shard_dim_idx_from_raw(0),
        shard_dim_idx_from_raw(1),
        shard_dim_idx_from_raw(2),
      };

      CHECK(result == correct);
    }

    SUBCASE("all dims have degree 1") {
      ParallelTensorDimDegrees degrees = ParallelTensorDimDegrees{
        SumDegree{1_p},
        DiscardCopyDegree{1_p},
        FFOrdered{1_p, 1_p, 1_p},
      };

      std::set<parallel_tensor_dim_idx_t> result = get_nontrivial_parallel_tensor_dim_indices(degrees);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }
  }
}
