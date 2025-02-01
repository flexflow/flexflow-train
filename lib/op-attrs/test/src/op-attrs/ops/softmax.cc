#include "op-attrs/ops/softmax.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/optional.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(SoftmaxAttrs, TensorShape)") {
    TensorShape input = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{
            12_n,
            14_n,
            16_n,
        }},
        DataType::FLOAT,
    };

    SUBCASE("attrs.dim in bounds") {
      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1_n}};

      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs, input);
      tl::expected<TensorShape, std::string> correct = input;

      CHECK(result == correct);
    }

    SUBCASE("attrs.dims out of bounds") {
      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{4_n}};

      std::optional<TensorShape> result =
          optional_from_expected(get_output_shape(attrs, input));
      std::optional<TensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }
  }

  TEST_CASE("get_output_shape(SoftmaxAttrs, ParallelTensorShape)") {
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

    SUBCASE("partition parallelism in non-softmax-dim (valid)") {
      nonnegative_int degree0 = 2_n;
      nonnegative_int degree2 = 4_n;

      ParallelTensorShape par_input = make_input(
          SumDegree{1_n}, DiscardCopyDegree{1_n}, degree0, 1_n, degree2);

      SUBCASE("attrs.dim in bounds") {
        SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1_n}};

        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct = make_output(
            SumDegree{1_n}, DiscardCopyDegree{1_n}, degree0, 1_n, degree2);

        CHECK(result == correct);
      }

      SUBCASE("attrs.dims out of bounds") {
        SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{4_n}};

        std::optional<ParallelTensorShape> result =
            optional_from_expected(get_output_shape(attrs, par_input));
        std::optional<ParallelTensorShape> correct = std::nullopt;

        CHECK(result == correct);
      }
    }

    SUBCASE("partition parallism in softmax dim (invalid)") {
      nonnegative_int degree1 = 2_n;

      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1_n}};

      ParallelTensorShape par_input =
          make_input(SumDegree{1_n}, DiscardCopyDegree{1_n}, 1_n, degree1, 1_n);

      std::optional<ParallelTensorShape> result =
          optional_from_expected(get_output_shape(attrs, par_input));
      std::optional<ParallelTensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("sum parallelism (invalid)") {
      SumDegree sum_degree = SumDegree{2_n};

      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1_n}};

      ParallelTensorShape par_input =
          make_input(sum_degree, DiscardCopyDegree{1_n}, 1_n, 1_n, 1_n);

      std::optional<ParallelTensorShape> result =
          optional_from_expected(get_output_shape(attrs, par_input));
      std::optional<ParallelTensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("discard copy parallelism (invalid)") {
      DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{2_n};

      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1_n}};

      ParallelTensorShape par_input =
          make_input(SumDegree{1_n}, discard_copy_degree, 1_n, 1_n, 1_n);

      std::optional<ParallelTensorShape> result =
          optional_from_expected(get_output_shape(attrs, par_input));
      std::optional<ParallelTensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
}
