#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "test/utils/doctest/fmt/map.h"
#include "test/utils/doctest/fmt/set.h"
#include <doctest/doctest.h>

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

    std::map<parallel_tensor_dim_idx_t, positive_int> result =
        get_parallel_tensor_degree_map(degrees);
    std::map<parallel_tensor_dim_idx_t, positive_int> correct = {
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

    std::set<ParallelTensorSpaceCoordinate> result =
        get_parallel_tensor_space_coordinates(degrees);
    std::set<ParallelTensorSpaceCoordinate> correct = {
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

  TEST_CASE(
      "get_nontrivial_parallel_tensor_dim_indices(ParallelTensorDimDegrees)") {
    SUBCASE("a replica dim has degree 1") {
      ParallelTensorDimDegrees degrees = ParallelTensorDimDegrees{
          SumDegree{3_p},
          DiscardCopyDegree{1_p},
          FFOrdered{4_p, 2_p, 4_p},
      };

      std::set<parallel_tensor_dim_idx_t> result =
          get_nontrivial_parallel_tensor_dim_indices(degrees);
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

      std::set<parallel_tensor_dim_idx_t> result =
          get_nontrivial_parallel_tensor_dim_indices(degrees);
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

      std::set<parallel_tensor_dim_idx_t> result =
          get_nontrivial_parallel_tensor_dim_indices(degrees);
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

      std::set<parallel_tensor_dim_idx_t> result =
          get_nontrivial_parallel_tensor_dim_indices(degrees);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }
  }

  TEST_CASE("parallel_dim_degrees_filter_shard_dims") {
    ParallelTensorDimDegrees input =
        ParallelTensorDimDegrees{SumDegree{5_p},
                                 DiscardCopyDegree{3_p},
                                 FFOrdered<positive_int>{
                                     2_p,
                                     1_p,
                                     1_p,
                                     4_p,
                                 }};

    SUBCASE("drop all shard dims") {
      ParallelTensorDimDegrees result = parallel_dim_degrees_filter_shard_dims(
          input, [](ff_dim_t d) -> bool { return false; });

      ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
          SumDegree{5_p},
          DiscardCopyDegree{3_p},
          FFOrdered<positive_int>{},
      };

      CHECK(result == correct);
    }

    SUBCASE("keep all shard dims") {
      ParallelTensorDimDegrees result = parallel_dim_degrees_filter_shard_dims(
          input, [](ff_dim_t d) -> bool { return true; });

      ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
          SumDegree{5_p},
          DiscardCopyDegree{3_p},
          FFOrdered<positive_int>{
              2_p,
              1_p,
              1_p,
              4_p,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("drop even dims") {
      ParallelTensorDimDegrees result = parallel_dim_degrees_filter_shard_dims(
          input, [](ff_dim_t d) -> bool { return d.value % 2_n == 1; });

      ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
          SumDegree{5_p},
          DiscardCopyDegree{3_p},
          FFOrdered<positive_int>{
              1_p,
              4_p,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("dim_domain_for_ptensor_dim_idxs") {
    ParallelTensorDimDegrees dim_degrees = ParallelTensorDimDegrees{
        SumDegree{2_p},
        DiscardCopyDegree{3_p},
        FFOrdered<positive_int>{
            5_p,
            7_p,
            11_p,
            13_p,
        },
    };

    SUBCASE("dim idx set is subset") {
      std::set<parallel_tensor_dim_idx_t> dim_idxs = {
          discard_copy_dim_idx(),
          shard_dim_idx(ff_dim_t{0_n}),
          shard_dim_idx(ff_dim_t{2_n}),
          shard_dim_idx(ff_dim_t{3_n}),
      };

      DimDomain<parallel_tensor_dim_idx_t> result =
          dim_domain_for_ptensor_dim_idxs(dim_degrees, dim_idxs);

      DimDomain<parallel_tensor_dim_idx_t> correct =
          DimDomain<parallel_tensor_dim_idx_t>{
              /*dims=*/std::map<parallel_tensor_dim_idx_t, positive_int>{
                  {discard_copy_dim_idx(), 3_p},
                  {shard_dim_idx(ff_dim_t{0_n}), 5_p},
                  {shard_dim_idx(ff_dim_t{2_n}), 11_p},
                  {shard_dim_idx(ff_dim_t{3_n}), 13_p},
              },
          };
    }

    SUBCASE("dim idx set contains other/invalid elements") {
      std::set<parallel_tensor_dim_idx_t> dim_idxs = {
          discard_copy_dim_idx(),
          shard_dim_idx(ff_dim_t{0_n}),
          shard_dim_idx(ff_dim_t{2_n}),
          shard_dim_idx(ff_dim_t{3_n}),
          shard_dim_idx(ff_dim_t{4_n}),
          shard_dim_idx(ff_dim_t{5_n}),
      };

      CHECK_THROWS(dim_domain_for_ptensor_dim_idxs(dim_degrees, dim_idxs));
    }
  }

  TEST_CASE("get_degree_for_relative_ff_dim_t") {
    ParallelTensorDimDegrees degrees = ParallelTensorDimDegrees{
        SumDegree{3_p},
        DiscardCopyDegree{7_p},
        FFOrdered<positive_int>{
            5_p,
            1_p,
            2_p,
            6_p,
        },
    };

    SUBCASE("dim = 0") {
      positive_int result =
          get_degree_for_relative_ff_dim_t(degrees, relative_ff_dim_t{0});

      positive_int correct = 5_p;

      CHECK(result == correct);
    }

    SUBCASE("dim = 1") {
      positive_int result =
          get_degree_for_relative_ff_dim_t(degrees, relative_ff_dim_t{1});

      positive_int correct = 1_p;

      CHECK(result == correct);
    }

    SUBCASE("dim = -1") {
      positive_int result =
          get_degree_for_relative_ff_dim_t(degrees, relative_ff_dim_t{-1});

      positive_int correct = 6_p;

      CHECK(result == correct);
    }

    SUBCASE("last positive dim (dim = 3)") {
      positive_int result =
          get_degree_for_relative_ff_dim_t(degrees, relative_ff_dim_t{3});

      positive_int correct = 6_p;

      CHECK(result == correct);
    }

    SUBCASE("first out of bounds positive dim (dim = 4)") {
      CHECK_THROWS(
          get_degree_for_relative_ff_dim_t(degrees, relative_ff_dim_t{4}));
    }

    SUBCASE("last negative dim (dim = -4)") {
      positive_int result =
          get_degree_for_relative_ff_dim_t(degrees, relative_ff_dim_t{-4});

      positive_int correct = 5_p;

      CHECK(result == correct);
    }

    SUBCASE("first out of bounds negative dim (dim = -5)") {
      CHECK_THROWS(
          get_degree_for_relative_ff_dim_t(degrees, relative_ff_dim_t{-5}));
    }
  }

  TEST_CASE("parallel_tensor_space_contains_coord") {
    ParallelTensorDimDegrees degrees = ParallelTensorDimDegrees{
        /*sum_degree=*/SumDegree{2_p},
        /*discard_copy_degree=*/DiscardCopyDegree{8_p},
        /*shard_degrees=*/
        FFOrdered<positive_int>{
            3_p,
            1_p,
            5_p,
        },
    };

    SUBCASE("degrees and coord do not have the same num dimensions") {
      ParallelTensorSpaceCoordinate coord = ParallelTensorSpaceCoordinate{
          /*sum_component=*/0_n,
          /*discard_copy_component=*/5_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              2_n,
              1_n,
          },
      };

      CHECK_THROWS(parallel_tensor_space_contains_coord(degrees, coord));
    }

    SUBCASE("coord is in bounds for degrees") {
      ParallelTensorSpaceCoordinate coord = ParallelTensorSpaceCoordinate{
          /*sum_component=*/0_n,
          /*discard_copy_component=*/5_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              2_n,
              0_n,
              1_n,
          },
      };

      bool result = parallel_tensor_space_contains_coord(degrees, coord);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("coord is out of bounds for degrees") {
      ParallelTensorSpaceCoordinate coord = ParallelTensorSpaceCoordinate{
          /*sum_component=*/0_n,
          /*discard_copy_component=*/5_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              3_n,
              0_n,
              1_n,
          },
      };

      bool result = parallel_tensor_space_contains_coord(degrees, coord);
      bool correct = false;

      CHECK(result == correct);
    }
  }
}
