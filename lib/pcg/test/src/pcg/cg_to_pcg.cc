#include "pcg/cg_to_pcg.h"
#include "op-attrs/activation.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "test/utils/rapidcheck.h"
#include "utils/containers/get_only.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("cg_to_pcg") {
    std::string input_name = "input";
    std::string dense_name = "dense";
    std::string relu_name = "relu";

    ComputationGraph cg = [&] {
      ComputationGraphBuilder b;

      TensorShape input_shape = TensorShape{
          TensorDims{FFOrdered<nonnegative_int>{10_n, 12_n}},
          DataType::FLOAT,
      };

      tensor_guid_t input =
          b.create_input(input_shape, CreateGrad::YES, input_name);

      tensor_guid_t dense = b.dense(input,
                                    /*outDim=*/8_n,
                                    /*activation=*/Activation::RELU,
                                    /*use_bias=*/true,
                                    /*data_type=*/DataType::FLOAT,
                                    /*projection_initializer=*/std::nullopt,
                                    /*bias_initializer=*/std::nullopt,
                                    /*name=*/dense_name);

      b.relu(dense, relu_name);

      return b.computation_graph;
    }();

    ParallelComputationGraph correct = [&] {
      ParallelComputationGraphBuilder b;

      ParallelTensorShape par_input_shape = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{10_n, 1_n},
                  ShardParallelDim{12_n, 1_n},
              },
              ReplicaParallelDimSet{
                  SumDegree{1_n},
                  DiscardCopyDegree{0_n},
              },
          },
          DataType::FLOAT,
      };

      parallel_tensor_guid_t input =
          b.create_input_tensor(par_input_shape, CreateGrad::YES, input_name);

      parallel_tensor_guid_t dense =
          b.dense(input,
                  /*outDim=*/8_n,
                  /*activation=*/Activation::RELU,
                  /*use_bias=*/true,
                  /*data_type=*/DataType::FLOAT,
                  /*projection_initializer=*/std::nullopt,
                  /*bias_initializer=*/std::nullopt,
                  /*name=*/dense_name);

      b.relu(dense, relu_name);

      return b.pcg;
    }();

    ParallelComputationGraph result = cg_to_pcg(cg);
    CHECK(pcgs_are_isomorphic(result, correct));
    CHECK(get_parallel_layers(result).size() ==
          get_parallel_layers(correct).size());
  }
}
