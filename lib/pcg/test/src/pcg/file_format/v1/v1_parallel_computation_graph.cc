#include "pcg/file_format/v1/v1_parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("V1ParallelComputationGraph") {
    ParallelComputationGraph pcg = [] {
      ParallelComputationGraphBuilder b;

      TensorShape input_shape = TensorShape{
          TensorDims{
              FFOrdered{
                  12_p,
                  16_p,
              },
          },
          DataType::FLOAT,
      };

      parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
      parallel_tensor_guid_t t_partition =
          b.parallel_partition(input, ff_dim_t{0_n}, 2_p);
      parallel_tensor_guid_t mm_output = b.dense(input, 8_p);
      parallel_tensor_guid_t relu_output = b.relu(mm_output);

      return b.pcg;
    }();

    V1ParallelComputationGraph v1_pcg = to_v1(pcg);
    nlohmann::json j = v1_pcg;
  }
}
