#include "op-attrs/ops/element_unary.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ReLU shape inference") {
    nonnegative_int d1 = 16_n;
    nonnegative_int d2 = 32_n;
    nonnegative_int d3 = 24_n;

    ElementUnaryAttrs attrs =
        ElementUnaryAttrs{OperatorType::RELU, std::nullopt};

    TensorShape input = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                d1,
                d2,
                d3,
            },
        },
        DataType::FLOAT,
    };

    tl::expected<TensorShape, std::string> result =
        get_output_shape(attrs, input);
    tl::expected<TensorShape, std::string> correct = input;

    CHECK(result == correct);

    auto make_input = [&](SumDegree o_sum,
                          DiscardCopyDegree o_eq,
                          nonnegative_int o_1,
                          nonnegative_int o_2,
                          nonnegative_int o_3) {
      return lift_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered<nonnegative_int>{o_1, o_2, o_3});
    };

    SUBCASE("partition i.e., sharding parallelism") {
      nonnegative_int degree1 = 4_n;
      nonnegative_int degree2 = 8_n;
      ParallelTensorShape par_input = make_input(
          SumDegree{1_n}, DiscardCopyDegree{1_n}, degree1, 1_n, degree2);

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, par_input);
      tl::expected<ParallelTensorShape, std::string> correct = par_input;

      CHECK(result == correct);
    }

    SUBCASE("sum degree > 1") {
      nonnegative_int degree = 2_n;

      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_input(SumDegree{degree}, DiscardCopyDegree{1_n}, 1_n, 1_n, 1_n));

      CHECK_MESSAGE(!result.has_value(),
                    "Unexpected successful result: ",
                    result.error());
    }

    SUBCASE("discard copy degree > 1") {
      nonnegative_int degree = 2_n;

      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_input(SumDegree{1_n}, DiscardCopyDegree{degree}, 1_n, 1_n, 1_n));

      CHECK_MESSAGE(!result.has_value(),
                    "Unexpected successful result: ",
                    result.error());
    }
  }
}
