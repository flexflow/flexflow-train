#include "op-attrs/ops/cast.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Cast shape inference") {
    DataType input_datatype = DataType::FLOAT;
    DataType output_datatype = DataType::DOUBLE;

    CastAttrs attrs = CastAttrs{output_datatype};

    nonnegative_int d1 = 12_n;
    nonnegative_int d2 = 16_n;
    TensorShape input = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{d1, d2}},
        input_datatype,
    };

    TensorShape output = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{d1, d2}},
        output_datatype,
    };

    SUBCASE("get_output_shape(CastAttrs, TensorShape)") {
      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs, input);
      tl::expected<TensorShape, std::string> correct = output;
      CHECK(result == correct);
    }

    SUBCASE("get_output_shape(CastAttrs, ParallelTensorShape)") {
      auto make_input = [&](SumDegree o_sum,
                            DiscardCopyDegree o_eq,
                            nonnegative_int o_batch,
                            nonnegative_int o_features) {
        return lift_to_parallel_with_degrees(
            input,
            o_sum,
            o_eq,
            FFOrdered<nonnegative_int>{o_batch, o_features});
      };

      auto make_output = [&](SumDegree o_sum,
                             DiscardCopyDegree o_eq,
                             nonnegative_int o_batch,
                             nonnegative_int o_outchannels) {
        return lift_to_parallel_with_degrees(
            output,
            o_sum,
            o_eq,
            FFOrdered<nonnegative_int>{o_batch, o_outchannels});
      };

      SumDegree sum_degree = SumDegree{2_n};
      DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{3_n};
      nonnegative_int batch_degree = 4_n;
      nonnegative_int feature_degree = 8_n;
      ParallelTensorShape par_input = make_input(
          sum_degree, discard_copy_degree, batch_degree, feature_degree);

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, par_input);
      tl::expected<ParallelTensorShape, std::string> correct = make_output(
          sum_degree, discard_copy_degree, batch_degree, feature_degree);

      CHECK(result == correct);
    }
  }
}
