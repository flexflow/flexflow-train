#include "compiler/graph_optimize_state.h"
#include "doctest/doctest.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("GraphOptimizeState::operator==") {
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                32_n,
                16_n,
            },
        },
        DataType::FLOAT,
    };
    // ParallelTensorShape input_shape =
    //     ParallelTensorShape{ParallelTensorDims{
    //                             FFOrdered<ShardParallelDim>{
    //                                 ShardParallelDim{32_n, 2_n},
    //                                 ShardParallelDim{16_n, 1_n},
    //                             },
    //                             ReplicaParallelDimSet{
    //                                 SumDegree{1_n},
    //                                 DiscardCopyDegree{1_n},
    //                             },
    //                         },
    //                         DataType::FLOAT};

    // `machine_mapping` is determined by the PCG and the device mapping
    // algorithm, and `runtime` is determined by the PCG and the device mapping,
    // so their values here do not matter.
    std::unordered_map<parallel_layer_guid_t, MachineView> empty_machine_views;
    MachineMapping empty_machine_mapping(empty_machine_views);

    InitializerAttrs zero_init = InitializerAttrs{ZeroInitializerAttrs{}};

    auto create_pcg = [&]() -> ParallelComputationGraph {
      ParallelComputationGraphBuilder builder;

      parallel_tensor_guid_t input0 =
          builder.create_input_tensor(input_shape, "input0");
      parallel_tensor_guid_t dense0 =
          builder.dense(/*input=*/input0,
                        /*outDim=*/8_n,
                        /*activation=*/Activation::RELU,
                        /*use_bias=*/true,
                        /*data_type=*/DataType::FLOAT,
                        /*projection_initializer=*/zero_init,
                        /*bias_initializer=*/zero_init,
                        /*name=*/"dense0");

      parallel_tensor_guid_t dense1 =
          builder.dense(/*input=*/dense0,
                        /*outDim=*/4_n,
                        /*activation=*/Activation::RELU,
                        /*use_bias=*/true,
                        /*data_type=*/DataType::FLOAT,
                        /*projection_initializer=*/zero_init,
                        /*bias_initializer=*/zero_init,
                        /*name=*/"dense1");

      return builder.pcg;
    };

    ParallelComputationGraph pcg1 = create_pcg();

    SUBCASE("returns true if the PCGs are isomorphic") {
      ParallelComputationGraph pcg2 = create_pcg();

      GraphOptimizeState state1 = GraphOptimizeState{
          GraphOptimizeResult{pcg1, empty_machine_mapping},
          0,
      };

      GraphOptimizeState state2 = GraphOptimizeState{
          GraphOptimizeResult{pcg2, empty_machine_mapping},
          0,
      };

      CHECK(state1 == state2);
    }

    SUBCASE("returns false it the PCGs are not isomorphic") {
      ParallelComputationGraphBuilder builder_;

      parallel_tensor_guid_t input0_ =
          builder_.create_input_tensor(input_shape, "input0");
      parallel_tensor_guid_t dense0_ =
          builder_.dense(/*input=*/input0_,
                         /*outDim=*/8_n,
                         /*activation=*/Activation::RELU,
                         /*use_bias=*/true,
                         /*data_type=*/DataType::FLOAT,
                         /*projection_initializer=*/zero_init,
                         /*bias_initializer=*/zero_init,
                         /*name=*/"dense0");

      ParallelComputationGraph pcg_ = builder_.pcg;

      GraphOptimizeState state1 = GraphOptimizeState{
          GraphOptimizeResult{pcg1, empty_machine_mapping},
          0,
      };

      GraphOptimizeState state_ = GraphOptimizeState{
          GraphOptimizeResult{pcg_, empty_machine_mapping},
          0,
      };

      CHECK_FALSE(state1 == state_);
    }
  }
}
