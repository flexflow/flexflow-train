#include "op-attrs/ops/batch_matmul.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(BatchMatmulAttrs, TensorShape)") {
    nonnegative_int b = 4_n;
    nonnegative_int m = 6_n;
    nonnegative_int n = 8_n;
    nonnegative_int p = 10_n;

    BatchMatmulAttrs attrs = BatchMatmulAttrs{
        /*a_seq_length_dim=*/0_n, // TODO figure out if these arguments are
                                  // still relevant
        /*b_seq_length_dim=*/0_n,
    };

    TensorShape input_lhs_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                b,
                n,
                m,
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("valid") {
      TensorShape input_rhs_shape = TensorShape{
          TensorDims{
              FFOrdered<nonnegative_int>{
                  b,
                  m,
                  p,
              },
          },
          DataType::FLOAT,
      };

      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs, input_lhs_shape, input_rhs_shape);

      tl::expected<TensorShape, std::string> correct_output_shape = TensorShape{
          TensorDims{
              FFOrdered<nonnegative_int>{
                  b,
                  n,
                  p,
              },
          },
          DataType::FLOAT,
      };

      CHECK(result == correct_output_shape);
    }

    SUBCASE("mismatched b") {
      TensorShape input_rhs_shape = TensorShape{
          TensorDims{
              FFOrdered<nonnegative_int>{
                  b + 1_n,
                  m,
                  p,
              },
          },
          DataType::FLOAT,
      };

      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs, input_lhs_shape, input_rhs_shape);

      CHECK(!result.has_value());
    }

    SUBCASE("mismatched m") {
      TensorShape input_rhs_shape = TensorShape{
          TensorDims{
              FFOrdered<nonnegative_int>{
                  b,
                  m + 1_n,
                  p,
              },
          },
          DataType::FLOAT,
      };

      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs, input_lhs_shape, input_rhs_shape);

      CHECK(!result.has_value());
    }
  }

  TEST_CASE("get_output_shape(BatchMatmulAttrs, ParallelTensorShape)") {
    nonnegative_int b = 2_n * 2_n;
    nonnegative_int o_b = 2_n;
    nonnegative_int m = 3_n * 3_n;
    nonnegative_int o_m = 3_n;
    nonnegative_int n = 5_n * 5_n;
    nonnegative_int o_n = 5_n;
    nonnegative_int p = 7_n * 7_n;
    nonnegative_int o_p = 7_n;
    nonnegative_int o_sum = 11_n;

    BatchMatmulAttrs attrs = BatchMatmulAttrs{
        /*a_seq_length_dim=*/0_n, // TODO figure out if these arguments are
                                  // still relevant
        /*b_seq_length_dim=*/0_n,
    };

    auto make_lhs = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        nonnegative_int o_b,
                        nonnegative_int o_n,
                        nonnegative_int o_m) {
      return ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{b, o_b},
                  ShardParallelDim{n, o_n},
                  ShardParallelDim{m, o_m},
              },
              ReplicaParallelDimSet{
                  o_sum,
                  o_eq,
              },
          },
          DataType::FLOAT,
      };
    };

    auto make_rhs = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        nonnegative_int o_b,
                        nonnegative_int o_m,
                        nonnegative_int o_p) {
      return ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{b, o_b},
                  ShardParallelDim{m, o_m},
                  ShardParallelDim{p, o_p},
              },
              ReplicaParallelDimSet{
                  o_sum,
                  o_eq,
              },
          },
          DataType::FLOAT,
      };
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           nonnegative_int o_b,
                           nonnegative_int o_n,
                           nonnegative_int o_p) {
      return ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{b, o_b},
                  ShardParallelDim{n, o_n},
                  ShardParallelDim{p, o_p},
              },
              ReplicaParallelDimSet{
                  o_sum,
                  o_eq,
              },
          },
          DataType::FLOAT,
      };
    };

    SUBCASE("data parallel") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{1_n}, DiscardCopyDegree{1_n}, o_b, 1_n, 1_n),
          make_rhs(SumDegree{1_n}, DiscardCopyDegree{1_n}, o_b, 1_n, 1_n));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{1_n}, DiscardCopyDegree{1_n}, o_b, 1_n, 1_n);

      CHECK(result == correct);
    }

    SUBCASE("n parallel") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{1_n}, DiscardCopyDegree{1_n}, 1_n, o_n, 1_n),
          make_rhs(SumDegree{1_n}, DiscardCopyDegree{o_n}, 1_n, 1_n, 1_n));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{1_n}, DiscardCopyDegree{1_n}, 1_n, o_n, 1_n);

      CHECK(result == correct);
    }

    SUBCASE("p parallel") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{1_n}, DiscardCopyDegree{o_p}, 1_n, 1_n, 1_n),
          make_rhs(SumDegree{1_n}, DiscardCopyDegree{1_n}, 1_n, 1_n, o_p));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{1_n}, DiscardCopyDegree{1_n}, 1_n, 1_n, o_p);

      CHECK(result == correct);
    }

    SUBCASE("reduction parallel") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{1_n}, DiscardCopyDegree{1_n}, 1_n, 1_n, o_m),
          make_rhs(SumDegree{1_n}, DiscardCopyDegree{1_n}, 1_n, o_m, 1_n));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{o_m}, DiscardCopyDegree{1_n}, 1_n, 1_n, 1_n);

      CHECK(result == correct);
    }

    SUBCASE("propagate reduction lhs") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{1_n}, 1_n, 1_n, 1_n),
          make_rhs(SumDegree{1_n}, DiscardCopyDegree{o_sum}, 1_n, 1_n, 1_n));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{o_sum}, DiscardCopyDegree{1_n}, 1_n, 1_n, 1_n);

      CHECK(result == correct);
    }

    SUBCASE("propagate reduction rhs") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{1_n}, DiscardCopyDegree{o_sum}, 1_n, 1_n, 1_n),
          make_rhs(SumDegree{o_sum}, DiscardCopyDegree{1_n}, 1_n, 1_n, 1_n));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{o_sum}, DiscardCopyDegree{1_n}, 1_n, 1_n, 1_n);

      CHECK(result == correct);
    }

    SUBCASE("reduction lhs & reduction rhs") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{o_sum}, 1_n, 1_n, 1_n),
          make_rhs(SumDegree{o_sum}, DiscardCopyDegree{o_sum}, 1_n, 1_n, 1_n));
      tl::expected<ParallelTensorShape, std::string> correct = make_output(
          SumDegree{o_sum * o_sum}, DiscardCopyDegree{1_n}, 1_n, 1_n, 1_n);

      CHECK(result == correct);
    }

    SUBCASE("reduction lhs & rhs (invalid)") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{1_n}, 1_n, 1_n, 1_n),
          make_rhs(SumDegree{o_sum}, DiscardCopyDegree{1_n}, 1_n, 1_n, 1_n));

      CHECK_MESSAGE(
          !result.has_value(), "Unexpected successful value: ", result);
    }

    SUBCASE("reduction lhs & n") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{1_n}, 1_n, o_n, 1_n),
          make_rhs(
              SumDegree{1_n}, DiscardCopyDegree{o_sum * o_n}, 1_n, 1_n, 1_n));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{o_sum}, DiscardCopyDegree{1_n}, 1_n, o_n, 1_n);

      CHECK(result == correct);
    }

    SUBCASE("reduction lhs & reduction rhs & n") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{o_sum}, 1_n, o_n, 1_n),
          make_rhs(
              SumDegree{o_sum}, DiscardCopyDegree{o_sum * o_n}, 1_n, 1_n, 1_n));
      tl::expected<ParallelTensorShape, std::string> correct = make_output(
          SumDegree{o_sum * o_sum}, DiscardCopyDegree{1_n}, 1_n, o_n, 1_n);

      CHECK(result == correct);
    }

    SUBCASE("reduction lhs & reduction rhs & n & m") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{o_sum}, 1_n, o_n, o_m),
          make_rhs(
              SumDegree{o_sum}, DiscardCopyDegree{o_sum * o_n}, 1_n, o_m, 1_n));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{o_sum * o_sum * o_m},
                      DiscardCopyDegree{1_n},
                      1_n,
                      o_n,
                      1_n);

      CHECK(result == correct);
    }
  }
}
