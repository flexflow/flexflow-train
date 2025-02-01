#include "op-attrs/ops/replicate.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Replicate shape inference") {
    ReplicateAttrs attrs = ReplicateAttrs{
        /*replicate_degree=*/4_n,
    };

    ParallelTensorShape input = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{10_n, 2_n},
                ShardParallelDim{12_n, 1_n},
                ShardParallelDim{14_n, 2_n},
                ShardParallelDim{16_n, 2_n},
            },
            ReplicaParallelDimSet{
                SumDegree{3_n},
                DiscardCopyDegree{2_n},
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape result = get_output_shape(attrs, input);

    ParallelTensorShape correct_output = input;
    correct_output.dims.replica_dims.discard_copy_degree =
        DiscardCopyDegree{8_n};

    CHECK(result == correct_output);
  }
}
