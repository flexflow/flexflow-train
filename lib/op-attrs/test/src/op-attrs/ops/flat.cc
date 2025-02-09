#include "op-attrs/ops/flat.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include "utils/fmt/optional.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(FlatAttrs, TensorShape)") {
    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{
            2_n,
            4_n,
            2_n,
            3_n,
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
          TensorDims{FFOrdered<nonnegative_int>{
              2_n * 4_n * 2_n * 3_n,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("flatten trailing dims") {
      FlatAttrs attrs = FlatAttrs{
          /*start_dim=*/ff_dim_t{nonnegative_int{2}},
          /*end_dim=*/ff_dim_t{nonnegative_int{4}},
      };

      TensorShape result = get_output_shape(attrs, input_shape);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered<nonnegative_int>{
              2_n,
              4_n,
              2_n * 3_n,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("flatten leading dims") {
      FlatAttrs attrs = FlatAttrs{
          /*start_dim=*/ff_dim_t{nonnegative_int{0}},
          /*end_dim=*/ff_dim_t{nonnegative_int{2}},
      };

      TensorShape result = get_output_shape(attrs, input_shape);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered<nonnegative_int>{
              2_n * 4_n,
              2_n,
              3_n,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("flatten middle dims") {
      FlatAttrs attrs = FlatAttrs{
          /*start_dim=*/ff_dim_t{nonnegative_int{1}},
          /*end_dim=*/ff_dim_t{nonnegative_int{3}},
      };

      TensorShape result = get_output_shape(attrs, input_shape);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered<nonnegative_int>{
              2_n,
              4_n * 2_n,
              3_n,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("flatten no dims (start_dim == end_dim)") {
      FlatAttrs attrs = FlatAttrs{
          /*start_dim=*/ff_dim_t{nonnegative_int{2}},
          /*end_dim=*/ff_dim_t{nonnegative_int{2}},
      };

      TensorShape result = get_output_shape(attrs, input_shape);
      TensorShape correct = input_shape;

      CHECK(result == correct);
    }

    SUBCASE("flatten no dims (start_dim < end_dim)") {
      FlatAttrs attrs = FlatAttrs{
          /*start_dim=*/ff_dim_t{nonnegative_int{2}},
          /*end_dim=*/ff_dim_t{nonnegative_int{1}},
      };

      TensorShape result = get_output_shape(attrs, input_shape);
      TensorShape correct = input_shape;

      CHECK(result == correct);
    }
  }

  TEST_CASE(
      "get_output_parallel_dim_degrees(FlatAttrs, ParallelTensorDimDegrees)") {
    FlatAttrs attrs = FlatAttrs{/*start_dim=*/ff_dim_t{nonnegative_int{1}},
                                /*end_dim=*/ff_dim_t{nonnegative_int{3}}};

    SUBCASE("allows shard parallelism in non-flattened dims") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_n},
          DiscardCopyDegree{1_n},
          FFOrdered<nonnegative_int>{2_n, 1_n, 1_n, 3_n},
      };

      tl::expected<ParallelTensorDimDegrees, std::string> result =
          get_output_parallel_dim_degrees(attrs, input);
      tl::expected<ParallelTensorDimDegrees, std::string> correct =
          ParallelTensorDimDegrees{
              SumDegree{1_n},
              DiscardCopyDegree{1_n},
              FFOrdered<nonnegative_int>{2_n, 1_n, 3_n},
          };

      CHECK(result == correct);
    }

    SUBCASE("does not allow shard parallelism in flattened dims") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_n},
          DiscardCopyDegree{1_n},
          FFOrdered<nonnegative_int>{1_n, 1_n, 2_n, 1_n},
      };

      std::optional<ParallelTensorDimDegrees> result =
          optional_from_expected(get_output_parallel_dim_degrees(attrs, input));
      std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("allows sum parallelism") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{2_n},
          DiscardCopyDegree{1_n},
          FFOrdered<nonnegative_int>{1_n, 1_n, 1_n, 1_n},
      };

      std::optional<ParallelTensorDimDegrees> result =
          optional_from_expected(get_output_parallel_dim_degrees(attrs, input));
      std::optional<ParallelTensorDimDegrees> correct =
          ParallelTensorDimDegrees{
              SumDegree{2_n},
              DiscardCopyDegree{1_n},
              FFOrdered<nonnegative_int>{1_n, 1_n, 1_n},
          };

      CHECK(result == correct);
    }

    SUBCASE("allows discard copy parallelism") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_n},
          DiscardCopyDegree{2_n},
          FFOrdered<nonnegative_int>{1_n, 1_n, 1_n, 1_n},
      };

      std::optional<ParallelTensorDimDegrees> result =
          optional_from_expected(get_output_parallel_dim_degrees(attrs, input));
      std::optional<ParallelTensorDimDegrees> correct =
          ParallelTensorDimDegrees{
              SumDegree{1_n},
              DiscardCopyDegree{2_n},
              FFOrdered<nonnegative_int>{1_n, 1_n, 1_n},
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
                ShardParallelDim{4_n, 2_n},
                ShardParallelDim{8_n, 1_n},
                ShardParallelDim{6_n, 1_n},
                ShardParallelDim{9_n, 3_n},
            },
            ReplicaParallelDimSet{
                SumDegree{7_n},
                DiscardCopyDegree{5_n},
            },
        },
        DataType::FLOAT,
    };

    FlatAttrs attrs = FlatAttrs{
        /*start_dim=*/ff_dim_t{nonnegative_int{1_n}},
        /*end_dim=*/ff_dim_t{nonnegative_int{3_n}},
    };

    tl::expected<ParallelTensorShape, std::string> result =
        get_output_shape(attrs, input_shape);
    tl::expected<ParallelTensorShape, std::string> correct =
        ParallelTensorShape{
            ParallelTensorDims{
                FFOrdered<ShardParallelDim>{
                    ShardParallelDim{4_n, 2_n},
                    ShardParallelDim{8_n * 6_n, 1_n},
                    ShardParallelDim{9_n, 3_n},
                },
                ReplicaParallelDimSet{
                    SumDegree{7_n},
                    DiscardCopyDegree{5_n},
                },
            },
            DataType::FLOAT,
        };

    CHECK(result == correct);
  }
}
