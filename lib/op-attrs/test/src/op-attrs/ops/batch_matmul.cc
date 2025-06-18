#include "op-attrs/ops/batch_matmul.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(BatchMatmulAttrs, TensorShape)") {
    positive_int b = 4_p;
    positive_int m = 6_p;
    positive_int n = 8_p;
    positive_int p = 10_p;

    BatchMatmulAttrs attrs = BatchMatmulAttrs{
        /*a_seq_length_dim=*/0_n, // TODO figure out if these arguments are
                                  // still relevant
        /*b_seq_length_dim=*/0_n,
    };

    TensorShape input_lhs_shape = TensorShape{
        TensorDims{
            FFOrdered{
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
              FFOrdered{
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
              FFOrdered{
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
              FFOrdered{
                  b + 1_p,
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
              FFOrdered{
                  b,
                  m + 1_p,
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
    positive_int b = 2_p * 2_p;
    positive_int o_b = 2_p;
    positive_int m = 3_p * 3_p;
    positive_int o_m = 3_p;
    positive_int n = 5_p * 5_p;
    positive_int o_n = 5_p;
    positive_int p = 7_p * 7_p;
    positive_int o_p = 7_p;
    positive_int o_sum = 11_p;

    BatchMatmulAttrs attrs = BatchMatmulAttrs{
        /*a_seq_length_dim=*/0_n, // TODO figure out if these arguments are
                                  // still relevant
        /*b_seq_length_dim=*/0_n,
    };

    auto make_lhs = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        positive_int o_b,
                        positive_int o_n,
                        positive_int o_m) {
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
                        positive_int o_b,
                        positive_int o_m,
                        positive_int o_p) {
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
                           positive_int o_b,
                           positive_int o_n,
                           positive_int o_p) {
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
          make_lhs(SumDegree{1_p}, DiscardCopyDegree{1_p}, o_b, 1_p, 1_p),
          make_rhs(SumDegree{1_p}, DiscardCopyDegree{1_p}, o_b, 1_p, 1_p));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{1_p}, DiscardCopyDegree{1_p}, o_b, 1_p, 1_p);

      CHECK(result == correct);
    }

    SUBCASE("n parallel") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, o_n, 1_p),
          make_rhs(SumDegree{1_p}, DiscardCopyDegree{o_n}, 1_p, 1_p, 1_p));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, o_n, 1_p);

      CHECK(result == correct);
    }

    SUBCASE("p parallel") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{1_p}, DiscardCopyDegree{o_p}, 1_p, 1_p, 1_p),
          make_rhs(SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, 1_p, o_p));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, 1_p, o_p);

      CHECK(result == correct);
    }

    SUBCASE("reduction parallel") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, 1_p, o_m),
          make_rhs(SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, o_m, 1_p));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{o_m}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p);

      CHECK(result == correct);
    }

    SUBCASE("propagate reduction lhs") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p),
          make_rhs(SumDegree{1_p}, DiscardCopyDegree{o_sum}, 1_p, 1_p, 1_p));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{o_sum}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p);

      CHECK(result == correct);
    }

    SUBCASE("propagate reduction rhs") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{1_p}, DiscardCopyDegree{o_sum}, 1_p, 1_p, 1_p),
          make_rhs(SumDegree{o_sum}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{o_sum}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p);

      CHECK(result == correct);
    }

    SUBCASE("reduction lhs & reduction rhs") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{o_sum}, 1_p, 1_p, 1_p),
          make_rhs(SumDegree{o_sum}, DiscardCopyDegree{o_sum}, 1_p, 1_p, 1_p));
      tl::expected<ParallelTensorShape, std::string> correct = make_output(
          SumDegree{o_sum * o_sum}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p);

      CHECK(result == correct);
    }

    SUBCASE("reduction lhs & rhs (invalid)") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p),
          make_rhs(SumDegree{o_sum}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p));

      CHECK_MESSAGE(
          !result.has_value(), "Unexpected successful value: ", result);
    }

    SUBCASE("reduction lhs & n") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{1_p}, 1_p, o_n, 1_p),
          make_rhs(
              SumDegree{1_p}, DiscardCopyDegree{o_sum * o_n}, 1_p, 1_p, 1_p));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{o_sum}, DiscardCopyDegree{1_p}, 1_p, o_n, 1_p);

      CHECK(result == correct);
    }

    SUBCASE("reduction lhs & reduction rhs & n") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{o_sum}, 1_p, o_n, 1_p),
          make_rhs(
              SumDegree{o_sum}, DiscardCopyDegree{o_sum * o_n}, 1_p, 1_p, 1_p));
      tl::expected<ParallelTensorShape, std::string> correct = make_output(
          SumDegree{o_sum * o_sum}, DiscardCopyDegree{1_p}, 1_p, o_n, 1_p);

      CHECK(result == correct);
    }

    SUBCASE("reduction lhs & reduction rhs & n & m") {
      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_lhs(SumDegree{o_sum}, DiscardCopyDegree{o_sum}, 1_p, o_n, o_m),
          make_rhs(
              SumDegree{o_sum}, DiscardCopyDegree{o_sum * o_n}, 1_p, o_m, 1_p));
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{o_sum * o_sum * o_m},
                      DiscardCopyDegree{1_p},
                      1_p,
                      o_n,
                      1_p);

      CHECK(result == correct);
    }
  }
}
