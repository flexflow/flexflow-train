#include "op-attrs/ops/element_unary.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ReLU shape inference") {
    positive_int d1 = 16_p;
    positive_int d2 = 32_p;
    positive_int d3 = 24_p;

    ElementUnaryAttrs attrs =
        ElementUnaryAttrs{OperatorType::RELU, std::nullopt};

    TensorShape input = TensorShape{
        TensorDims{
            FFOrdered{
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
                          positive_int o_1,
                          positive_int o_2,
                          positive_int o_3) {
      return lift_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered{o_1, o_2, o_3});
    };

    SUBCASE("partition i.e., sharding parallelism") {
      positive_int degree1 = 4_p;
      positive_int degree2 = 8_p;
      ParallelTensorShape par_input = make_input(
          SumDegree{1_p}, DiscardCopyDegree{1_p}, degree1, 1_p, degree2);

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, par_input);
      tl::expected<ParallelTensorShape, std::string> correct = par_input;

      CHECK(result == correct);
    }

    SUBCASE("sum degree > 1") {
      positive_int degree = 2_p;

      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_input(SumDegree{degree}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p));

      CHECK_MESSAGE(!result.has_value(),
                    "Unexpected successful result: ",
                    result.error());
    }

    SUBCASE("discard copy degree > 1") {
      positive_int degree = 2_p;

      tl::expected<ParallelTensorShape, std::string> result = get_output_shape(
          attrs,
          make_input(SumDegree{1_p}, DiscardCopyDegree{degree}, 1_p, 1_p, 1_p));

      CHECK_MESSAGE(!result.has_value(),
                    "Unexpected successful result: ",
                    result.error());
    }
  }
}
