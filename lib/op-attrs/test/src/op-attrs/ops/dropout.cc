#include "op-attrs/ops/dropout.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/optional.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(DropoutAttrs, TensorShape)") {
    DropoutAttrs attrs = DropoutAttrs{
        /*rate=*/0.5,
        /*seed=*/1,
    };

    TensorShape input = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{
            12_n,
            14_n,
            16_n,
        }},
        DataType::FLOAT,
    };

    TensorShape result = get_output_shape(attrs, input);
    TensorShape correct = input;

    CHECK(result == correct);
  }

  TEST_CASE("get_output_shape(DropoutAttrs, ParallelTensorShape)") {
    DropoutAttrs attrs = DropoutAttrs{
        /*rate=*/0.5,
        /*seed=*/1,
    };

    TensorShape input = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{
            12_n,
            14_n,
            16_n,
        }},
        DataType::FLOAT,
    };

    TensorShape output = input;

    auto make_input = [&](SumDegree o_sum,
                          DiscardCopyDegree o_eq,
                          nonnegative_int o0,
                          nonnegative_int o1,
                          nonnegative_int o2) {
      return lift_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered<nonnegative_int>{o0, o1, o2});
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           nonnegative_int o0,
                           nonnegative_int o1,
                           nonnegative_int o2) {
      return lift_to_parallel_with_degrees(
          output, o_sum, o_eq, FFOrdered<nonnegative_int>{o0, o1, o2});
    };

    SUBCASE("partition parallelism (allowed)") {
      nonnegative_int degree0 = 2_n;
      nonnegative_int degree2 = 4_n;

      ParallelTensorShape par_input = make_input(
          SumDegree{1_n}, DiscardCopyDegree{1_n}, degree0, 1_n, degree2);

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, par_input);
      tl::expected<ParallelTensorShape, std::string> correct = make_output(
          SumDegree{1_n}, DiscardCopyDegree{1_n}, degree0, 1_n, degree2);

      CHECK(result == correct);
    }

    SUBCASE("sum parallelism (not allowed)") {
      SumDegree sum_degree = SumDegree{2_n};

      ParallelTensorShape par_input =
          make_input(sum_degree, DiscardCopyDegree{1_n}, 1_n, 1_n, 1_n);

      std::optional<ParallelTensorShape> result =
          optional_from_expected(get_output_shape(attrs, par_input));
      std::optional<ParallelTensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("discard copy parallelism (not allowed)") {
      DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{2_n};

      ParallelTensorShape par_input =
          make_input(SumDegree{1_n}, discard_copy_degree, 1_n, 1_n, 1_n);

      std::optional<ParallelTensorShape> result =
          optional_from_expected(get_output_shape(attrs, par_input));
      std::optional<ParallelTensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
}
