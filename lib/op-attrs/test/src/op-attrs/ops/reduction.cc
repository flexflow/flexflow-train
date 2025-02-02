#include "op-attrs/ops/reduction.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Reduction shape inference") {

    ParallelTensorShape input = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{12_n, 2_n},
                ShardParallelDim{14_n, 1_n},
                ShardParallelDim{16_n, 3_n},
                ShardParallelDim{18_n, 2_n},
            },
            ReplicaParallelDimSet{
                SumDegree{3_n},
                DiscardCopyDegree{2_n},
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("valid") {
      nonnegative_int degree = 3_n;
      ReductionAttrs attrs = ReductionAttrs{
          /*repartition_degree=*/degree,
      };

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, input);

      tl::expected<ParallelTensorShape, std::string> correct = [&] {
        ParallelTensorShape output = input;
        output.dims.replica_dims.sum_degree.value /= degree;
        return output;
      }();

      CHECK(result == correct);
    }

    SUBCASE("invalid") {
      nonnegative_int degree = 4_n;
      ReductionAttrs attrs = ReductionAttrs{
          /*repartition_degree=*/degree,
      };

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, input);

      CHECK_MESSAGE(!result.has_value(),
                    "Unexpected successful result: ",
                    result.error());
    }
  }
}
