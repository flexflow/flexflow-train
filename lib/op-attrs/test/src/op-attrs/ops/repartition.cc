#include "op-attrs/ops/repartition.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Repartition shape inference") {
    ff_dim_t dim = ff_dim_t{2_n};
    nonnegative_int degree = 4_n;
    RepartitionAttrs attrs = RepartitionAttrs{
        /*repartition_dim=*/dim,
        /*repartition_degree=*/degree,
    };

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
