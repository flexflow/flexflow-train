#include "pcg/file_format/v1/v1_computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("V1ComputationGraph") {
    ComputationGraph cg = [] {
      ComputationGraphBuilder b;

      TensorShape input_shape = TensorShape{
          TensorDims{FFOrdered<nonnegative_int>{
              12_n,
              16_n,
          }},
          DataType::FLOAT,
      };

      tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES);
      tensor_guid_t mm_output = b.dense(input, 8_n);
      tensor_guid_t relu_output = b.relu(mm_output);

      return b.computation_graph;
    }();

    V1ComputationGraph v1_cg = to_v1(cg);
    nlohmann::json j = v1_cg;
  }
}
