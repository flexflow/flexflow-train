#include "op-attrs/ops/element_binary.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("EWAdd shape inference") {
    nonnegative_int d1 = 16_n;
    nonnegative_int d2 = 32_n;
    nonnegative_int d3 = 24_n;

    ElementBinaryAttrs attrs = ElementBinaryAttrs{
        OperatorType::EW_ADD,
        DataType::FLOAT,
        /*should_broadcast_lhs=*/false,
        /*should_broadcast_rhs=*/false,
    };

    TensorShape input_lhs = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                d1,
                d2,
                d3,
            },
        },
        DataType::FLOAT,
    };

    TensorShape input_rhs = input_lhs;

    SUBCASE("correct") {
      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs, input_lhs, input_rhs);
      tl::expected<TensorShape, std::string> correct = input_lhs;

      CHECK(result == correct);
    }

    SUBCASE("mismatched dim size") {
      TensorShape incorrect_rhs = input_lhs;
      dim_at_idx(incorrect_rhs, relative_ff_dim_t{0}) += 1_n;

      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs, input_lhs, incorrect_rhs);

      CHECK_MESSAGE(!result.has_value(),
                    "Unexpected successful result: ",
                    result.error());
    }
  }

  TEST_CASE("EWAdd parallel shape inference") {
    nonnegative_int d1 = 16_n;
    nonnegative_int d2 = 32_n;
    nonnegative_int d3 = 24_n;

    ElementBinaryAttrs attrs = ElementBinaryAttrs{
        OperatorType::EW_ADD,
        DataType::FLOAT,
        /*should_broadcast_lhs=*/false,
        /*should_broadcast_rhs=*/false,
    };

    TensorShape unpar_lhs = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                d1,
                d2,
                d3,
            },
        },
        DataType::FLOAT,
    };

    TensorShape unpar_rhs = unpar_lhs;
    tl::expected<TensorShape, std::string> result_unpar_output =
        get_output_shape(attrs, unpar_lhs, unpar_rhs);
    REQUIRE(result_unpar_output.has_value());
    TensorShape unpar_output = result_unpar_output.value();

    auto make_lhs = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        nonnegative_int o_1,
                        nonnegative_int o_2,
                        nonnegative_int o_3) {
      return lift_to_parallel_with_degrees(
          unpar_lhs, o_sum, o_eq, FFOrdered<nonnegative_int>{o_1, o_2, o_3});
    };

    auto make_rhs = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        nonnegative_int o_1,
                        nonnegative_int o_2,
                        nonnegative_int o_3) {
      return lift_to_parallel_with_degrees(
          unpar_rhs, o_sum, o_eq, FFOrdered<nonnegative_int>{o_1, o_2, o_3});
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           nonnegative_int o_1,
                           nonnegative_int o_2,
                           nonnegative_int o_3) {
      return lift_to_parallel_with_degrees(
          unpar_output, o_sum, o_eq, FFOrdered<nonnegative_int>{o_1, o_2, o_3});
    };

    SUBCASE("data parallelism") {
      nonnegative_int degree = 4_n;

      ParallelTensorShape input_lhs =
          make_lhs(SumDegree{1_n}, DiscardCopyDegree{1_n}, degree, 1_n, 1_n);
      ParallelTensorShape input_rhs =
          make_rhs(SumDegree{1_n}, DiscardCopyDegree{1_n}, degree, 1_n, 1_n);
      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, input_lhs, input_rhs);
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{1_n}, DiscardCopyDegree{1_n}, degree, 1_n, 1_n);

      CHECK(result == correct);
    }

    SUBCASE("reduction parallelism") {
      nonnegative_int degree = 4_n;

      ParallelTensorShape input_lhs =
          make_lhs(SumDegree{degree}, DiscardCopyDegree{1_n}, 1_n, 1_n, 1_n);
      ParallelTensorShape input_rhs =
          make_rhs(SumDegree{degree}, DiscardCopyDegree{1_n}, 1_n, 1_n, 1_n);
      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, input_lhs, input_rhs);
      tl::expected<ParallelTensorShape, std::string> correct =
          make_output(SumDegree{degree}, DiscardCopyDegree{1_n}, 1_n, 1_n, 1_n);

      CHECK(result == correct);
    }

    SUBCASE("invalid discard copy parallelism") {
      nonnegative_int degree = 4_n;

      ParallelTensorShape input_lhs =
          make_lhs(SumDegree{1_n}, DiscardCopyDegree{degree}, 1_n, 1_n, 1_n);
      ParallelTensorShape input_rhs =
          make_rhs(SumDegree{1_n}, DiscardCopyDegree{degree}, 1_n, 1_n, 1_n);
      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, input_lhs, input_rhs);

      CHECK_MESSAGE(!result.has_value(),
                    "Unexpected successful result: ",
                    result.error());
    }

    SUBCASE("invalid mismatched parallelism degrees") {
      nonnegative_int degree = 4_n;

      ParallelTensorShape input_lhs =
          make_lhs(SumDegree{1_n}, DiscardCopyDegree{1_n}, 1_n, degree, 1_n);
      ParallelTensorShape input_rhs =
          make_rhs(SumDegree{1_n}, DiscardCopyDegree{1_n}, 1_n, 1_n, degree);
      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, input_lhs, input_rhs);

      CHECK_MESSAGE(!result.has_value(),
                    "Unexpected successful result: ",
                    result.error());
    }
  }
}
