#include "op-attrs/ops/repartition.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Repartition shape inference") {
    ff_dim_t dim = ff_dim_t{2_n};
    positive_int degree = 4_p;
    RepartitionAttrs attrs = RepartitionAttrs{
        /*repartition_dim=*/dim,
        /*repartition_degree=*/degree,
    };

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

    tl::expected<ParallelTensorShape, std::string> result =
        get_output_shape(attrs, input);

    tl::expected<ParallelTensorShape, std::string> correct = [&] {
      ParallelTensorShape output = input;
      output.dims.shard_dims.at(dim).degree *= degree;
      return output;
    }();

    CHECK(result == correct);
  }
}
