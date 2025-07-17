#include "op-attrs/ops/flat.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include "utils/fmt/optional.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(FlatAttrs, TensorShape)") {
    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{
            2_p,
            4_p,
            2_p,
            3_p,
        }},
        DataType::FLOAT,
    };

    SUBCASE("flatten all dims") {
      FlatAttrs attrs = FlatAttrs{
          /*start_dim=*/ff_dim_t{0_n},
          /*end_dim=*/ff_dim_t{4_n},
      };

      TensorShape result = get_output_shape(attrs, input_shape);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{
              2_p * 4_p * 2_p * 3_p,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("flatten trailing dims") {
      FlatAttrs attrs = FlatAttrs{
          /*start_dim=*/ff_dim_t{2_n},
          /*end_dim=*/ff_dim_t{4_n},
      };

      TensorShape result = get_output_shape(attrs, input_shape);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{
              2_p,
              4_p,
              2_p * 3_p,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("flatten leading dims") {
      FlatAttrs attrs = FlatAttrs{
          /*start_dim=*/ff_dim_t{0_n},
          /*end_dim=*/ff_dim_t{2_n},
      };

      TensorShape result = get_output_shape(attrs, input_shape);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{
              2_p * 4_p,
              2_p,
              3_p,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("flatten middle dims") {
      FlatAttrs attrs = FlatAttrs{
          /*start_dim=*/ff_dim_t{1_n},
          /*end_dim=*/ff_dim_t{3_n},
      };

      TensorShape result = get_output_shape(attrs, input_shape);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{
              2_p,
              4_p * 2_p,
              3_p,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("flatten no dims (start_dim == end_dim)") {
      FlatAttrs attrs = FlatAttrs{
          /*start_dim=*/ff_dim_t{2_n},
          /*end_dim=*/ff_dim_t{2_n},
      };

      TensorShape result = get_output_shape(attrs, input_shape);
      TensorShape correct = input_shape;

      CHECK(result == correct);
    }

    SUBCASE("flatten no dims (start_dim < end_dim)") {
      FlatAttrs attrs = FlatAttrs{
          /*start_dim=*/ff_dim_t{2_n},
          /*end_dim=*/ff_dim_t{1_n},
      };

      TensorShape result = get_output_shape(attrs, input_shape);
      TensorShape correct = input_shape;

      CHECK(result == correct);
    }
  }

  TEST_CASE(
      "get_output_parallel_dim_degrees(FlatAttrs, ParallelTensorDimDegrees)") {
    FlatAttrs attrs = FlatAttrs{/*start_dim=*/ff_dim_t{1_n},
                                /*end_dim=*/ff_dim_t{3_n}};

    SUBCASE("allows shard parallelism in non-flattened dims") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{2_p, 1_p, 1_p, 3_p},
      };

      tl::expected<ParallelTensorDimDegrees, std::string> result =
          get_output_parallel_dim_degrees(attrs, input);
      tl::expected<ParallelTensorDimDegrees, std::string> correct =
          ParallelTensorDimDegrees{
              SumDegree{1_p},
              DiscardCopyDegree{1_p},
              FFOrdered{2_p, 1_p, 3_p},
          };

      CHECK(result == correct);
    }

    SUBCASE("does not allow shard parallelism in flattened dims") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{1_p, 1_p, 2_p, 1_p},
      };

      std::optional<ParallelTensorDimDegrees> result =
          optional_from_expected(get_output_parallel_dim_degrees(attrs, input));
      std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("allows sum parallelism") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{2_p},
          DiscardCopyDegree{1_p},
          FFOrdered{1_p, 1_p, 1_p, 1_p},
      };

      std::optional<ParallelTensorDimDegrees> result =
          optional_from_expected(get_output_parallel_dim_degrees(attrs, input));
      std::optional<ParallelTensorDimDegrees> correct =
          ParallelTensorDimDegrees{
              SumDegree{2_p},
              DiscardCopyDegree{1_p},
              FFOrdered{1_p, 1_p, 1_p},
          };

      CHECK(result == correct);
    }

    SUBCASE("allows discard copy parallelism") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{2_p},
          FFOrdered{1_p, 1_p, 1_p, 1_p},
      };

      std::optional<ParallelTensorDimDegrees> result =
          optional_from_expected(get_output_parallel_dim_degrees(attrs, input));
      std::optional<ParallelTensorDimDegrees> correct =
          ParallelTensorDimDegrees{
              SumDegree{1_p},
              DiscardCopyDegree{2_p},
              FFOrdered{1_p, 1_p, 1_p},
          };

      CHECK(result == correct);
    }
  }

  TEST_CASE("get_output_shape(FlatAttrs, ParallelTensorShape)") {
    // since most of the edge cases are already tested in
    // get_output_shape(FlatAttrs, TensorShape) and
    // get_output_parallel_dim_degrees(FlatAttrs, ParallelTensorDimDegrees),
    // here we just do a basic check that they compose

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{4_p, 2_p},
                ShardParallelDim{8_p, 1_p},
                ShardParallelDim{6_p, 1_p},
                ShardParallelDim{9_p, 3_p},
            },
            ReplicaParallelDimSet{
                SumDegree{7_p},
                DiscardCopyDegree{5_p},
            },
        },
        DataType::FLOAT,
    };

    FlatAttrs attrs = FlatAttrs{
        /*start_dim=*/ff_dim_t{nonnegative_int{1_p}},
        /*end_dim=*/ff_dim_t{nonnegative_int{3_p}},
    };

    tl::expected<ParallelTensorShape, std::string> result =
        get_output_shape(attrs, input_shape);
    tl::expected<ParallelTensorShape, std::string> correct =
        ParallelTensorShape{
            ParallelTensorDims{
                FFOrdered<ShardParallelDim>{
                    ShardParallelDim{4_p, 2_p},
                    ShardParallelDim{8_p * 6_p, 1_p},
                    ShardParallelDim{9_p, 3_p},
                },
                ReplicaParallelDimSet{
                    SumDegree{7_p},
                    DiscardCopyDegree{5_p},
                },
            },
            DataType::FLOAT,
        };

    CHECK(result == correct);
  }
}
