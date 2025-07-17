#include "op-attrs/ops/combine.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Combine shape inference") {

    ParallelTensorShape input = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{12_p, 2_p},
                ShardParallelDim{14_p, 1_p},
                ShardParallelDim{16_p, 3_p},
                ShardParallelDim{18_p, 2_p},
            },
            ReplicaParallelDimSet{
                SumDegree{3_p},
                DiscardCopyDegree{2_p},
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("valid") {
      ff_dim_t dim = ff_dim_t{2_n};
      positive_int degree = 3_p;
      CombineAttrs attrs = CombineAttrs{
          /*repartition_dim=*/dim,
          /*repartition_degree=*/degree,
      };

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, input);

      tl::expected<ParallelTensorShape, std::string> correct = [&] {
        ParallelTensorShape output = input;
        positive_int old_shard_degree = output.dims.shard_dims.at(dim).degree;
        output.dims.shard_dims.at(dim).degree =
            positive_int{old_shard_degree / degree};
        return output;
      }();

      CHECK(result == correct);
    }

    SUBCASE("invalid") {
      ff_dim_t dim = ff_dim_t{2_n};
      positive_int degree = 4_p;
      CombineAttrs attrs = CombineAttrs{
          /*repartition_dim=*/dim,
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
