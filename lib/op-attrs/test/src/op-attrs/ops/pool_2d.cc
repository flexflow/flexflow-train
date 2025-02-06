#include "op-attrs/ops/pool_2d.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include "utils/fmt/optional.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("make_adaptive_pool2d") {
    nonnegative_int input_n = 10_n;
    nonnegative_int input_c = 11_n;
    nonnegative_int input_h = 15_n;
    nonnegative_int input_w = 20_n;
    Activation activation = Activation::RELU;
    PoolOp op = PoolOp::AVG;

    TensorDims input_dims = TensorDims{
        FFOrdered<nonnegative_int>{input_n, input_c, input_h, input_w}};

    SUBCASE("input_h divisible by output_h && input_w divisible by output_w") {
      nonnegative_int output_h = 5_n;
      nonnegative_int output_w = 2_n;

      Pool2DAttrs correct_attrs = Pool2DAttrs{
          /*kernel_h=*/3_n,
          /*kernel_w=*/10_n,
          /*stride_h=*/3_n,
          /*stride_w=*/10_n,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*pool_type=*/op,
          /*activation=*/activation,
      };

      SUBCASE("returns correct attrs") {
        tl::expected<Pool2DAttrs, std::string> result =
            make_adaptive_pool2d_attrs(
                input_dims, output_h, output_w, op, activation);
        tl::expected<Pool2DAttrs, std::string> correct = correct_attrs;

        CHECK(result == correct);
      }

      SUBCASE(
          "confirm that output shape is as expected for the expected attrs") {
        TensorShape input_shape = TensorShape{input_dims, DataType::FLOAT};

        tl::expected<TensorShape, std::string> result =
            get_output_shape(correct_attrs, input_shape);
        tl::expected<TensorShape, std::string> correct = TensorShape{
            TensorDims{FFOrdered<nonnegative_int>{
                input_n,
                input_c,
                output_h,
                output_w,
            }},
            DataType::FLOAT,
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("input_h not divisible by output_h") {
      nonnegative_int output_h = 6_n;
      nonnegative_int output_w = 2_n;

      std::optional<Pool2DAttrs> result =
          optional_from_expected(make_adaptive_pool2d_attrs(
              input_dims, output_h, output_w, op, activation));
      std::optional<Pool2DAttrs> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("input_w not divisible by output_w") {
      nonnegative_int output_h = 5_n;
      nonnegative_int output_w = 3_n;

      std::optional<Pool2DAttrs> result =
          optional_from_expected(make_adaptive_pool2d_attrs(
              input_dims, output_h, output_w, op, activation));
      std::optional<Pool2DAttrs> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("input_h == output_h and input_w == output_w") {
      nonnegative_int output_h = input_h;
      nonnegative_int output_w = input_w;

      Pool2DAttrs correct_attrs = Pool2DAttrs{
          /*kernel_h=*/1_n,
          /*kernel_w=*/1_n,
          /*stride_h=*/1_n,
          /*stride_w=*/1_n,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*pool_type=*/op,
          /*activation=*/activation,
      };

      SUBCASE("returns correct attrs") {
        tl::expected<Pool2DAttrs, std::string> result =
            make_adaptive_pool2d_attrs(
                input_dims, output_h, output_w, op, activation);
        tl::expected<Pool2DAttrs, std::string> correct = correct_attrs;

        CHECK(result == correct);
      }

      SUBCASE(
          "confirm that output shape is as expected for the expected attrs") {
        TensorShape input_shape = TensorShape{input_dims, DataType::FLOAT};

        tl::expected<TensorShape, std::string> result =
            get_output_shape(correct_attrs, input_shape);
        tl::expected<TensorShape, std::string> correct = input_shape;

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("get_output_shape(Pool2DAttrs, TensorShape)") {
    Pool2DAttrs attrs = Pool2DAttrs{
        /*kernel_h=*/3_n,
        /*kernel_w=*/2_n,
        /*stride_h=*/2_n,
        /*stride_w=*/2_n,
        /*padding_h=*/1_n,
        /*padding_w=*/1_n,
        /*pool_type=*/PoolOp::MAX,
        /*activation=*/std::nullopt,
    };

    SUBCASE("fails on non-4d inputs") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<nonnegative_int>{
              10_n,
              12_n,
              14_n,
          }},
          DataType::FLOAT,
      };

      std::optional<TensorShape> result =
          optional_from_expected(get_output_shape(attrs, input));
      std::optional<TensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("4d input") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<nonnegative_int>{11_n, 13_n, 12_n, 6_n}},
          DataType::FLOAT,
      };

      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs, input);
      tl::expected<TensorShape, std::string> correct = TensorShape{
          TensorDims{FFOrdered<nonnegative_int>{11_n, 13_n, 6_n, 4_n}},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("get_output_parallel_dim_degrees(Pool2DAttrs, "
            "ParallelTensorDimDegrees)") {
    auto make_attrs = [](PoolOp pool_type,
                         std::optional<Activation> const &activation) {
      return Pool2DAttrs{
          /*kernel_h=*/3_n,
          /*kernel_w=*/2_n,
          /*stride_h=*/2_n,
          /*stride_w=*/2_n,
          /*padding_h=*/1_n,
          /*padding_w=*/1_n,
          /*pool_type=*/pool_type,
          /*activation=*/activation,
      };
    };

    SUBCASE("allows data parallelism") {
      Pool2DAttrs attrs = make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_n},
          DiscardCopyDegree{1_n},
          FFOrdered<nonnegative_int>{
              4_n,
              1_n,
              1_n,
              1_n,
          },
      };

      tl::expected<ParallelTensorDimDegrees, std::string> result =
          get_output_parallel_dim_degrees(attrs, input);
      tl::expected<ParallelTensorDimDegrees, std::string> correct = input;

      CHECK(result == correct);
    }

    SUBCASE("allows arbitrary input sharding parallelism") {
      Pool2DAttrs attrs = make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_n},
          DiscardCopyDegree{1_n},
          FFOrdered<nonnegative_int>{
              4_n,
              2_n,
              5_n,
              6_n,
          },
      };

      tl::expected<ParallelTensorDimDegrees, std::string> result =
          get_output_parallel_dim_degrees(attrs, input);
      tl::expected<ParallelTensorDimDegrees, std::string> correct = input;

      CHECK(result == correct);
    }

    SUBCASE("allows discard copy parallelism") {
      Pool2DAttrs attrs = make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_n},
          DiscardCopyDegree{3_n},
          FFOrdered<nonnegative_int>{
              1_n,
              1_n,
              1_n,
              1_n,
          },
      };

      tl::expected<ParallelTensorDimDegrees, std::string> result =
          get_output_parallel_dim_degrees(attrs, input);
      tl::expected<ParallelTensorDimDegrees, std::string> correct = input;

      CHECK(result == correct);
    }

    SUBCASE("sum parallelism") {
      SUBCASE("without activation") {
        SUBCASE("PoolOp::MAX does not allow sum parallelism") {
          Pool2DAttrs attrs =
              make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

          ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
              SumDegree{2_n},
              DiscardCopyDegree{1_n},
              FFOrdered<nonnegative_int>{
                  1_n,
                  1_n,
                  1_n,
                  1_n,
              },
          };

          std::optional<ParallelTensorDimDegrees> result =
              optional_from_expected(
                  get_output_parallel_dim_degrees(attrs, input));
          std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

          CHECK(result == correct);
        }

        SUBCASE("PoolOp::AVG does allow sum parallelism") {
          Pool2DAttrs attrs =
              make_attrs(PoolOp::AVG, /*activation=*/std::nullopt);

          ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
              SumDegree{2_n},
              DiscardCopyDegree{1_n},
              FFOrdered<nonnegative_int>{
                  1_n,
                  1_n,
                  1_n,
                  1_n,
              },
          };

          tl::expected<ParallelTensorDimDegrees, std::string> result =
              get_output_parallel_dim_degrees(attrs, input);
          tl::expected<ParallelTensorDimDegrees, std::string> correct = input;

          CHECK(result == correct);
        }
      }

      SUBCASE("with activation does not allow sum parallelism") {
        Pool2DAttrs attrs =
            make_attrs(PoolOp::AVG, /*activation=*/Activation::RELU);

        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{2_n},
            DiscardCopyDegree{1_n},
            FFOrdered<nonnegative_int>{
                1_n,
                1_n,
                1_n,
                1_n,
            },
        };

        std::optional<ParallelTensorDimDegrees> result = optional_from_expected(
            get_output_parallel_dim_degrees(attrs, input));
        std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("get_output_shape(Pool2DAttrs, ParallelTensorShape)") {
    // this function is mostly covered by the tests above, so we
    // just do a single test to make sure it works/exists

    Pool2DAttrs attrs = Pool2DAttrs{
        /*kernel_h=*/3_n,
        /*kernel_w=*/2_n,
        /*stride_h=*/2_n,
        /*stride_w=*/2_n,
        /*padding_h=*/1_n,
        /*padding_w=*/1_n,
        /*pool_type=*/PoolOp::MAX,
        /*activation=*/std::nullopt,
    };

    SUBCASE("valid parallelism") {
      ParallelTensorShape input = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{14_n, 7_n},
                  ShardParallelDim{16_n, 8_n},
                  ShardParallelDim{12_n, 3_n},
                  ShardParallelDim{6_n, 2_n},
              },
              ReplicaParallelDimSet{
                  SumDegree{1_n},
                  DiscardCopyDegree{2_n},
              },
          },
          DataType::FLOAT,
      };

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, input);
      tl::expected<ParallelTensorShape, std::string> correct =
          ParallelTensorShape{
              ParallelTensorDims{
                  FFOrdered<ShardParallelDim>{
                      ShardParallelDim{14_n, 7_n},
                      ShardParallelDim{16_n, 8_n},
                      ShardParallelDim{6_n, 3_n},
                      ShardParallelDim{4_n, 2_n},
                  },
                  ReplicaParallelDimSet{
                      SumDegree{1_n},
                      DiscardCopyDegree{2_n},
                  },
              },
              DataType::FLOAT,
          };
    }

    SUBCASE("invalid parallelism") {
      ParallelTensorShape input = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{14_n, 1_n},
                  ShardParallelDim{16_n, 1_n},
                  ShardParallelDim{12_n, 1_n},
                  ShardParallelDim{6_n, 1_n},
              },
              ReplicaParallelDimSet{
                  SumDegree{2_n},
                  DiscardCopyDegree{1_n},
              },
          },
          DataType::FLOAT,
      };

      std::optional<ParallelTensorShape> result =
          optional_from_expected(get_output_shape(attrs, input));
      std::optional<ParallelTensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
}
