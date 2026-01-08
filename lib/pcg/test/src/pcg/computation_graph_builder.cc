#include "pcg/computation_graph_builder.h"
#include "pcg/computation_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ComputationGraphBuilder") {
    ComputationGraphBuilder b;

    positive_int batch_size = 2_p;

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, 3_p, 10_p, 10_p}},
        DataType::FLOAT,
    };

    tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES);
    tensor_guid_t output = b.conv2d(input,
                                    /*outChannels=*/5_p,
                                    /*kernelH=*/3_p,
                                    /*kernelW=*/3_p,
                                    /*strideH=*/1_p,
                                    /*strideW=*/1_p,
                                    /*paddingH=*/0_n,
                                    /*paddingW=*/0_n);
    // ComputationGraph cg = b.computation_graph;
    // CHECK(get_layers(cg).size() == 1);
  }
}
