#include "pcg/computation_graph_builder.h"
#include "doctest/doctest.h"
#include "pcg/computation_graph.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ComputationGraphBuilder") {
    ComputationGraphBuilder b;

    nonnegative_int batch_size = 2_n;

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{batch_size, 3_n, 10_n, 10_n}},
        DataType::FLOAT,
    };

    tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES);
    tensor_guid_t output = b.conv2d(input,
                                    /*outChannels=*/5_n,
                                    /*kernelH=*/3_n,
                                    /*kernelW=*/3_n,
                                    /*strideH=*/1_n,
                                    /*strideW=*/1_n,
                                    /*paddingH=*/0_n,
                                    /*paddingW=*/0_n);
    // ComputationGraph cg = b.computation_graph;
    // CHECK(get_layers(cg).size() == 1);
  }
}
