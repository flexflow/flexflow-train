#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "test/utils/doctest/fmt/optional.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

static ParallelTensorSpaceCoordinate mk_0d_coord(int sum_component,
                                                 int discard_copy_component) {
  return ParallelTensorSpaceCoordinate{
      /*sum_component=*/nonnegative_int{sum_component},
      /*discard_copy_component=*/nonnegative_int{discard_copy_component},
      /*shard_components=*/FFOrdered<nonnegative_int>{},
  };
}

static ParallelTensorSpaceCoordinate mk_1d_coord(int sum_component,
                                                 int discard_copy_component,
                                                 int shard_0_component) {
  return ParallelTensorSpaceCoordinate{
      /*sum_component=*/nonnegative_int{sum_component},
      /*discard_copy_component=*/nonnegative_int{discard_copy_component},
      /*shard_components=*/
      FFOrdered<nonnegative_int>{
          nonnegative_int{shard_0_component},
      },
  };
}

static ParallelTensorSpaceCoordinate mk_2d_coord(int sum_component,
                                                 int discard_copy_component,
                                                 int shard_0_component,
                                                 int shard_1_component) {
  return ParallelTensorSpaceCoordinate{
      /*sum_component=*/nonnegative_int{sum_component},
      /*discard_copy_component=*/nonnegative_int{discard_copy_component},
      /*shard_components=*/
      FFOrdered<nonnegative_int>{
          nonnegative_int{shard_0_component},
          nonnegative_int{shard_1_component},
      },
  };
}

static ParallelTensorSpaceCoordinate mk_3d_coord(int sum_component,
                                                 int discard_copy_component,
                                                 int shard_0_component,
                                                 int shard_1_component,
                                                 int shard_2_component) {
  return ParallelTensorSpaceCoordinate{
      /*sum_component=*/nonnegative_int{sum_component},
      /*discard_copy_component=*/nonnegative_int{discard_copy_component},
      /*shard_components=*/
      FFOrdered<nonnegative_int>{
          nonnegative_int{shard_0_component},
          nonnegative_int{shard_1_component},
          nonnegative_int{shard_2_component},
      },
  };
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("smallest_parallel_tensor_dim_degrees_for_coord_set") {
    SUBCASE("coord set is empty") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {};

      CHECK_THROWS(
          smallest_parallel_tensor_dim_degrees_for_coord_set(coord_set));
    }

    SUBCASE("coord set contains coords of different dimensionalities") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(1, 2, 0),
          mk_2d_coord(0, 4, 0, 3),
      };

      CHECK_THROWS(
          smallest_parallel_tensor_dim_degrees_for_coord_set(coord_set));
    }

    SUBCASE("coord set is not orthotopic") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 1, 2),
          mk_1d_coord(2, 4, 1),
          mk_1d_coord(3, 0, 0),
      };

      std::optional<ParallelTensorDimDegrees> result =
          smallest_parallel_tensor_dim_degrees_for_coord_set(coord_set);
      std::optional<ParallelTensorDimDegrees> correct =
          ParallelTensorDimDegrees{
              /*sum_degrees=*/SumDegree{4_p},
              /*discard_copy_degrees=*/DiscardCopyDegree{5_p},
              /*shard_degrees=*/
              FFOrdered<positive_int>{
                  3_p,
              },
          };

      CHECK(result == correct);
    }

    SUBCASE("coord set is orthotopic") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 0, 0),
          mk_1d_coord(0, 1, 0),
          mk_1d_coord(0, 2, 0),
          mk_1d_coord(1, 0, 0),
          mk_1d_coord(1, 1, 0),
          mk_1d_coord(1, 2, 0),
          mk_1d_coord(0, 0, 1),
          mk_1d_coord(0, 1, 1),
          mk_1d_coord(0, 2, 1),
          mk_1d_coord(1, 0, 1),
          mk_1d_coord(1, 1, 1),
          mk_1d_coord(1, 2, 1),
      };

      std::optional<ParallelTensorDimDegrees> result =
          smallest_parallel_tensor_dim_degrees_for_coord_set(coord_set);
      std::optional<ParallelTensorDimDegrees> correct =
          ParallelTensorDimDegrees{
              /*sum_degree=*/SumDegree{2_p},
              /*discard_copy_degree=*/DiscardCopyDegree{3_p},
              /*shard_degrees=*/
              FFOrdered<positive_int>{
                  2_p,
              },
          };

      CHECK(result == correct);
    }

    SUBCASE("coord set's space is not minimal") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 0, 0),
          mk_1d_coord(0, 1, 0),
          mk_1d_coord(0, 2, 0),
          mk_1d_coord(1, 0, 0),
          mk_1d_coord(1, 1, 0),
          mk_1d_coord(1, 2, 0),
      };

      std::optional<ParallelTensorDimDegrees> result =
          smallest_parallel_tensor_dim_degrees_for_coord_set(coord_set);
      std::optional<ParallelTensorDimDegrees> correct =
          ParallelTensorDimDegrees{
              /*sum_degree=*/SumDegree{2_p},
              /*discard_copy_degree=*/DiscardCopyDegree{3_p},
              /*shard_degrees=*/
              FFOrdered<positive_int>{
                  1_p,
              },
          };
    }
  }

  TEST_CASE("strict_parallel_tensor_dim_degrees_for_coord_set") {
    SUBCASE("coord set is empty") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {};

      std::optional<ParallelTensorDimDegrees> result =
          strict_parallel_tensor_dim_degrees_for_coord_set(coord_set);
      std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("coord set contains coords of different dimensionalities") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(1, 2, 0),
          mk_2d_coord(0, 4, 0, 3),
      };

      CHECK_THROWS(strict_parallel_tensor_dim_degrees_for_coord_set(coord_set));
    }

    SUBCASE("coord set is not orthotopic") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 1, 2),
          mk_1d_coord(2, 4, 1),
          mk_1d_coord(3, 0, 0),
      };

      std::optional<ParallelTensorDimDegrees> result =
          strict_parallel_tensor_dim_degrees_for_coord_set(coord_set);
      std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("coord set is orthotopic") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 0, 0),
          mk_1d_coord(0, 1, 0),
          mk_1d_coord(0, 2, 0),
          mk_1d_coord(1, 0, 0),
          mk_1d_coord(1, 1, 0),
          mk_1d_coord(1, 2, 0),
          mk_1d_coord(0, 0, 1),
          mk_1d_coord(0, 1, 1),
          mk_1d_coord(0, 2, 1),
          mk_1d_coord(1, 0, 1),
          mk_1d_coord(1, 1, 1),
          mk_1d_coord(1, 2, 1),
      };

      std::optional<ParallelTensorDimDegrees> result =
          strict_parallel_tensor_dim_degrees_for_coord_set(coord_set);
      std::optional<ParallelTensorDimDegrees> correct =
          ParallelTensorDimDegrees{
              /*sum_degree=*/SumDegree{2_p},
              /*discard_copy_degree=*/DiscardCopyDegree{3_p},
              /*shard_degrees=*/
              FFOrdered<positive_int>{
                  2_p,
              },
          };

      CHECK(result == correct);
    }

    SUBCASE("coord set's space is not minimal") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 0, 0),
          mk_1d_coord(0, 1, 0),
          mk_1d_coord(0, 2, 0),
          mk_1d_coord(1, 0, 0),
          mk_1d_coord(1, 1, 0),
          mk_1d_coord(1, 2, 0),
      };

      std::optional<ParallelTensorDimDegrees> result =
          strict_parallel_tensor_dim_degrees_for_coord_set(coord_set);
      std::optional<ParallelTensorDimDegrees> correct =
          ParallelTensorDimDegrees{
              /*sum_degree=*/SumDegree{2_p},
              /*discard_copy_degree=*/DiscardCopyDegree{3_p},
              /*shard_degrees=*/
              FFOrdered<positive_int>{
                  1_p,
              },
          };
    }
  }

  TEST_CASE("parallel_tensor_coord_set_is_orthotopic") {
    SUBCASE("coord set is empty") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {};

      bool result = parallel_tensor_coord_set_is_orthotopic(coord_set);
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("coord set contains coords of different dimensionalities") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(1, 2, 0),
          mk_2d_coord(0, 4, 0, 3),
      };

      CHECK_THROWS(parallel_tensor_coord_set_is_orthotopic(coord_set));
    }

    SUBCASE("coord set is not orthotopic") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 1, 2),
          mk_1d_coord(2, 4, 1),
          mk_1d_coord(3, 0, 0),
      };

      bool result = parallel_tensor_coord_set_is_orthotopic(coord_set);
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("coord set is orthotopic") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 0, 0),
          mk_1d_coord(0, 1, 0),
          mk_1d_coord(0, 2, 0),
          mk_1d_coord(1, 0, 0),
          mk_1d_coord(1, 1, 0),
          mk_1d_coord(1, 2, 0),
          mk_1d_coord(0, 0, 1),
          mk_1d_coord(0, 1, 1),
          mk_1d_coord(0, 2, 1),
          mk_1d_coord(1, 0, 1),
          mk_1d_coord(1, 1, 1),
          mk_1d_coord(1, 2, 1),
      };

      bool result = parallel_tensor_coord_set_is_orthotopic(coord_set);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("coord set's space is orthotopic but not minimal") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 0, 0),
          mk_1d_coord(0, 1, 0),
          mk_1d_coord(0, 2, 0),
          mk_1d_coord(1, 0, 0),
          mk_1d_coord(1, 1, 0),
          mk_1d_coord(1, 2, 0),
      };

      bool result = parallel_tensor_coord_set_is_orthotopic(coord_set);
      bool correct = true;

      CHECK(result == correct);
    }
  }

  TEST_CASE(
      "parallel_tensor_space_coordinate_from_bounded_orthotope_components") {
    BoundedComponent sum_component = BoundedComponent{1_n, 8_p};
    BoundedComponent discard_copy_component = BoundedComponent{2_n, 3_p};
    OrthotopeBoundedCoord shard_components =
        make_orthotope_bounded_coord_from_components({
            BoundedComponent{1_n, 5_p},
            BoundedComponent{0_n, 1_p},
            BoundedComponent{3_n, 6_p},
            BoundedComponent{0_n, 4_p},
        });

    ParallelTensorSpaceCoordinate result =
        parallel_tensor_space_coordinate_from_bounded_orthotope_components(
            sum_component, discard_copy_component, shard_components);

    ParallelTensorSpaceCoordinate correct = ParallelTensorSpaceCoordinate{
        /*sum_component=*/1_n,
        /*discard_copy_component=*/2_n,
        /*shard_components=*/
        FFOrdered<nonnegative_int>{
            1_n,
            0_n,
            3_n,
            0_n,
        },
    };

    CHECK(result == correct);
  }

  TEST_CASE("orthotope_bounded_coord_for_ptensor_dims") {
    ParallelTensorDimDegrees dim_degrees = ParallelTensorDimDegrees{
        /*sum_degree=*/SumDegree{8_p},
        /*discard_copy_degree=*/DiscardCopyDegree{3_p},
        /*shard_degrees=*/
        FFOrdered<positive_int>{
            5_p,
            1_p,
            6_p,
            4_p,
        },
    };

    SUBCASE("degrees and coord do not have same number of dimensions") {
      ParallelTensorSpaceCoordinate coord = ParallelTensorSpaceCoordinate{
          /*sum_component=*/1_n,
          /*discard_copy_component=*/2_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              2_n,
              0_n,
              4_n,
          },
      };

      std::set<parallel_tensor_dim_idx_t> desired_dims = {
          shard_dim_idx(ff_dim_t{0_n}), sum_dim_idx()};

      CHECK_THROWS(orthotope_bounded_coord_for_ptensor_dims(
          dim_degrees, coord, desired_dims));
    }

    SUBCASE("coord is out of bounds") {
      ParallelTensorSpaceCoordinate coord = ParallelTensorSpaceCoordinate{
          /*sum_component=*/1_n,
          /*discard_copy_component=*/2_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              2_n,
              0_n,
              6_n,
              0_n,
          },
      };

      std::set<parallel_tensor_dim_idx_t> desired_dims = {
          shard_dim_idx(ff_dim_t{0_n}), sum_dim_idx()};

      CHECK_THROWS(orthotope_bounded_coord_for_ptensor_dims(
          dim_degrees, coord, desired_dims));
    }

    SUBCASE("desired dims is out of bounds") {
      ParallelTensorSpaceCoordinate coord = ParallelTensorSpaceCoordinate{
          /*sum_component=*/1_n,
          /*discard_copy_component=*/2_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              2_n,
              0_n,
              4_n,
              0_n,
          },
      };

      std::set<parallel_tensor_dim_idx_t> desired_dims = {
          shard_dim_idx(ff_dim_t{4_n}), sum_dim_idx()};

      CHECK_THROWS(orthotope_bounded_coord_for_ptensor_dims(
          dim_degrees, coord, desired_dims));
    }

    SUBCASE("correct usage") {
      ParallelTensorSpaceCoordinate coord = ParallelTensorSpaceCoordinate{
          /*sum_component=*/1_n,
          /*discard_copy_component=*/2_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              2_n,
              0_n,
              4_n,
              0_n,
          },
      };

      std::set<parallel_tensor_dim_idx_t> desired_dims = {
          shard_dim_idx(ff_dim_t{0_n}), sum_dim_idx()};

      OrthotopeBoundedCoord result = orthotope_bounded_coord_for_ptensor_dims(
          dim_degrees, coord, desired_dims);
      OrthotopeBoundedCoord correct = OrthotopeBoundedCoord{
          /*coord=*/OrthotopeCoord{{1_n, 2_n}},
          /*bounds=*/
          Orthotope{
              {8_p, 5_p},
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("desired dims is empty") {
      ParallelTensorSpaceCoordinate coord = ParallelTensorSpaceCoordinate{
          /*sum_component=*/1_n,
          /*discard_copy_component=*/2_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              2_n,
              0_n,
              4_n,
              0_n,
          },
      };

      std::set<parallel_tensor_dim_idx_t> desired_dims = {};

      OrthotopeBoundedCoord result = orthotope_bounded_coord_for_ptensor_dims(
          dim_degrees, coord, desired_dims);
      OrthotopeBoundedCoord correct = OrthotopeBoundedCoord{
          /*coord=*/OrthotopeCoord{{}},
          /*bounds=*/Orthotope{{}},
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("bounded_component_for_ptensor_dim") {
    ParallelTensorDimDegrees dim_degrees = ParallelTensorDimDegrees{
        /*sum_degree=*/SumDegree{4_p},
        /*discard_copy_degree=*/DiscardCopyDegree{3_p},
        /*shard_degrees=*/
        FFOrdered<positive_int>{
            5_p,
            1_p,
            6_p,
            4_p,
        },
    };

    SUBCASE("degrees and coord do not have same number of dimensions") {
      ParallelTensorSpaceCoordinate coord = ParallelTensorSpaceCoordinate{
          /*sum_component=*/1_n,
          /*discard_copy_component=*/2_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              2_n,
              0_n,
              4_n,
          },
      };

      parallel_tensor_dim_idx_t desired_dim = shard_dim_idx(ff_dim_t{0_n});

      CHECK_THROWS(
          bounded_component_for_ptensor_dim(dim_degrees, coord, desired_dim));
    }

    SUBCASE("coord is out of bounds") {
      ParallelTensorSpaceCoordinate coord = ParallelTensorSpaceCoordinate{
          /*sum_component=*/1_n,
          /*discard_copy_component=*/2_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              2_n,
              0_n,
              6_n,
              0_n,
          },
      };

      parallel_tensor_dim_idx_t desired_dim = shard_dim_idx(ff_dim_t{0_n});

      CHECK_THROWS(
          bounded_component_for_ptensor_dim(dim_degrees, coord, desired_dim));
    }

    SUBCASE("desired dim is out of bounds") {
      ParallelTensorSpaceCoordinate coord = ParallelTensorSpaceCoordinate{
          /*sum_component=*/1_n,
          /*discard_copy_component=*/2_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              2_n,
              0_n,
              4_n,
              0_n,
          },
      };

      parallel_tensor_dim_idx_t desired_dim = shard_dim_idx(ff_dim_t{4_n});

      CHECK_THROWS(
          bounded_component_for_ptensor_dim(dim_degrees, coord, desired_dim));
    }

    SUBCASE("correct usage") {
      ParallelTensorSpaceCoordinate coord = ParallelTensorSpaceCoordinate{
          /*sum_component=*/1_n,
          /*discard_copy_component=*/2_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              2_n,
              0_n,
              4_n,
              0_n,
          },
      };

      parallel_tensor_dim_idx_t desired_dim = shard_dim_idx(ff_dim_t{0_n});

      BoundedComponent result =
          bounded_component_for_ptensor_dim(dim_degrees, coord, desired_dim);
      BoundedComponent correct = BoundedComponent{
          /*component=*/2_n,
          /*bound=*/5_p,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("parallel_tensor_space_coordinate_restrict_dims") {
    ParallelTensorSpaceCoordinate coord = ParallelTensorSpaceCoordinate{
        /*sum_component=*/2_n,
        /*discard_copy_component=*/0_n,
        /*shard_components=*/
        FFOrdered<nonnegative_int>{
            3_n,
            5_n,
        },
    };

    SUBCASE("allowed_dims is a superset of the coord's dims") {
      std::set<ff_dim_t> allowed_dims = {
          ff_dim_t{0_n},
          ff_dim_t{1_n},
          ff_dim_t{2_n},
          ff_dim_t{3_n},
      };

      ParallelTensorSpaceCoordinate result =
          parallel_tensor_space_coordinate_restrict_dims(coord, allowed_dims);

      ParallelTensorSpaceCoordinate correct = coord;

      CHECK(result == correct);
    }

    SUBCASE("allowed_dims is the same as the coord's dims") {
      std::set<ff_dim_t> allowed_dims = {
          ff_dim_t{0_n},
          ff_dim_t{1_n},
      };

      ParallelTensorSpaceCoordinate result =
          parallel_tensor_space_coordinate_restrict_dims(coord, allowed_dims);

      ParallelTensorSpaceCoordinate correct = coord;

      CHECK(result == correct);
    }

    SUBCASE("allowed_dims overlaps the coord's dims") {
      std::set<ff_dim_t> allowed_dims = {
          ff_dim_t{1_n},
          ff_dim_t{3_n},
      };

      ParallelTensorSpaceCoordinate result =
          parallel_tensor_space_coordinate_restrict_dims(coord, allowed_dims);

      ParallelTensorSpaceCoordinate correct =
          ParallelTensorSpaceCoordinate{/*sum_component=*/2_n,
                                        /*discard_copy_component=*/0_n,
                                        /*shard_compoonent=*/
                                        FFOrdered<nonnegative_int>{
                                            5_n,
                                        }};

      CHECK(result == correct);
    }

    SUBCASE("allowed_dims is a subset of the coord's dims") {
      std::set<ff_dim_t> allowed_dims = {ff_dim_t{0_n}};

      ParallelTensorSpaceCoordinate result =
          parallel_tensor_space_coordinate_restrict_dims(coord, allowed_dims);

      ParallelTensorSpaceCoordinate correct = ParallelTensorSpaceCoordinate{
          /*sum_component=*/2_n,
          /*discard_copy_component=*/0_n,
          /*shard_compoonent=*/
          FFOrdered<nonnegative_int>{
              3_n,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("allowed_dims is empty") {
      std::set<ff_dim_t> allowed_dims = {};

      ParallelTensorSpaceCoordinate result =
          parallel_tensor_space_coordinate_restrict_dims(coord, allowed_dims);

      ParallelTensorSpaceCoordinate correct = ParallelTensorSpaceCoordinate{
          /*sum_component=*/2_n,
          /*discard_copy_component=*/0_n,
          /*shard_compoonent=*/FFOrdered<nonnegative_int>{},
      };

      CHECK(result == correct);
    }

    SUBCASE("coord is empty") {
      std::set<ff_dim_t> allowed_dims = {
          ff_dim_t{0_n},
          ff_dim_t{1_n},
      };

      ParallelTensorSpaceCoordinate empty_coord = ParallelTensorSpaceCoordinate{
          /*sum_component=*/1_n,
          /*discard_copy_component=*/3_n,
          /*shard_compoonent=*/FFOrdered<nonnegative_int>{},
      };

      ParallelTensorSpaceCoordinate result =
          parallel_tensor_space_coordinate_restrict_dims(empty_coord,
                                                         allowed_dims);

      ParallelTensorSpaceCoordinate correct = empty_coord;

      CHECK(result == correct);
    }
  }

  TEST_CASE("parallel_tensor_space_coordinate_without_dims") {
    ParallelTensorSpaceCoordinate coord = ParallelTensorSpaceCoordinate{
        /*sum_component=*/1_n,
        /*discard_copy_component=*/3_n,
        /*shard_compoonent=*/
        FFOrdered<nonnegative_int>{
            4_n,
            0_n,
            2_n,
            1_n,
        },
    };

    SUBCASE("coord contains all of dim set") {
      std::set<ff_dim_t> dim_set = {
          ff_dim_t{0_n},
          ff_dim_t{2_n},
      };

      ParallelTensorSpaceCoordinate result =
          parallel_tensor_space_coordinate_without_dims(coord, dim_set);
      ParallelTensorSpaceCoordinate correct = ParallelTensorSpaceCoordinate{
          /*sum_component=*/1_n,
          /*discard_copy_component=*/3_n,
          /*shard_compoonent=*/
          FFOrdered<nonnegative_int>{
              0_n,
              1_n,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("dim set is all of coord") {
      std::set<ff_dim_t> dim_set = {
          ff_dim_t{0_n},
          ff_dim_t{1_n},
          ff_dim_t{2_n},
          ff_dim_t{3_n},
      };

      ParallelTensorSpaceCoordinate result =
          parallel_tensor_space_coordinate_without_dims(coord, dim_set);
      ParallelTensorSpaceCoordinate correct = ParallelTensorSpaceCoordinate{
          /*sum_component=*/1_n,
          /*discard_copy_component=*/3_n,
          /*shard_compoonent=*/FFOrdered<nonnegative_int>{},
      };

      CHECK(result == correct);
    }

    SUBCASE("coord does not contain all of dim set") {
      std::set<ff_dim_t> dim_set = {
          ff_dim_t{2_n},
          ff_dim_t{5_n},
      };

      CHECK_THROWS(
          parallel_tensor_space_coordinate_without_dims(coord, dim_set));
    }
  }
}
