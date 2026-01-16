#include "compiler/graph_optimize_state.h"
#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/machine_mapping/machine_view.dtg.h"
#include "compiler/machine_mapping/machine_view.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "test/utils/doctest/check_without_stringify.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("GraphOptimizeState operator==") {
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                32_p,
                16_p,
            },
        },
        DataType::FLOAT,
    };

    InitializerAttrs zero_init = InitializerAttrs{ZeroInitializerAttrs{}};

    auto create_pcg = [&]() -> ParallelComputationGraph {
      ParallelComputationGraphBuilder builder;

      parallel_tensor_guid_t input0 =
          builder.create_input_tensor(input_shape, "input0");
      parallel_tensor_guid_t dense0 =
          builder.dense(/*input=*/input0,
                        /*outDim=*/8_p,
                        /*activation=*/Activation::RELU,
                        /*use_bias=*/true,
                        /*data_type=*/DataType::FLOAT,
                        /*projection_initializer=*/zero_init,
                        /*bias_initializer=*/zero_init,
                        /*name=*/"dense0");

      parallel_tensor_guid_t dense1 =
          builder.dense(/*input=*/dense0,
                        /*outDim=*/4_p,
                        /*activation=*/Activation::RELU,
                        /*use_bias=*/true,
                        /*data_type=*/DataType::FLOAT,
                        /*projection_initializer=*/zero_init,
                        /*bias_initializer=*/zero_init,
                        /*name=*/"dense1");

      return builder.pcg;
    };

    auto create_machine_mapping_for_pcg =
        [](ParallelComputationGraph const &pcg) -> MachineMapping {
      MachineSpaceCoordinate device = MachineSpaceCoordinate{
          /*node_idx=*/0_n,
          /*device_idx=*/0_n,
          /*device_type=*/DeviceType::GPU,
      };

      MachineView machine_view = make_single_device_machine_view(device);

      return MachineMapping{
          generate_map(get_parallel_layers(pcg),
                       [&](parallel_layer_guid_t) { return machine_view; }),
      };
    };

    ParallelComputationGraph pcg1 = create_pcg();
    MachineMapping machine_mapping_1 = create_machine_mapping_for_pcg(pcg1);

    SUBCASE("returns true if the PCGs are isomorphic") {
      ParallelComputationGraph pcg2 = create_pcg();
      MachineMapping machine_mapping_2 = create_machine_mapping_for_pcg(pcg2);

      GraphOptimizeState state1 = GraphOptimizeState{
          GraphOptimizeResult{
              mapped_pcg_from_pcg_and_mapping(pcg1, machine_mapping_1),
          },
          0,
      };

      GraphOptimizeState state2 = GraphOptimizeState{
          GraphOptimizeResult{
              mapped_pcg_from_pcg_and_mapping(pcg2, machine_mapping_2),
          },
          0,
      };

      CHECK_WITHOUT_STRINGIFY(state1 == state2);
    }

    SUBCASE("returns false it the PCGs are not isomorphic") {
      ParallelComputationGraphBuilder builder_;

      parallel_tensor_guid_t input0_ =
          builder_.create_input_tensor(input_shape, "input0");
      parallel_tensor_guid_t dense0_ =
          builder_.dense(/*input=*/input0_,
                         /*outDim=*/8_p,
                         /*activation=*/Activation::RELU,
                         /*use_bias=*/true,
                         /*data_type=*/DataType::FLOAT,
                         /*projection_initializer=*/zero_init,
                         /*bias_initializer=*/zero_init,
                         /*name=*/"dense0");

      ParallelComputationGraph other_pcg = builder_.pcg;

      MachineMapping other_machine_mapping =
          create_machine_mapping_for_pcg(other_pcg);

      GraphOptimizeState state1 = GraphOptimizeState{
          GraphOptimizeResult{
              mapped_pcg_from_pcg_and_mapping(pcg1, machine_mapping_1),
          },
          0,
      };

      GraphOptimizeState state_ = GraphOptimizeState{
          GraphOptimizeResult{
              mapped_pcg_from_pcg_and_mapping(other_pcg, other_machine_mapping),
          },
          0,
      };

      CHECK_FALSE_WITHOUT_STRINGIFY(state1 == state_);
    }
  }
}
