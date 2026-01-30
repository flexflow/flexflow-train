#include "compiler/unity_algorithm/graph_optimize_state.h"
#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/machine_mapping/machine_view.dtg.h"
#include "compiler/machine_mapping/machine_view.h"
#include "doctest/doctest.h"
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

    ParallelComputationGraph pcg1 = create_pcg();

    SUBCASE("returns true if the PCGs are isomorphic") {
      ParallelComputationGraph pcg2 = create_pcg();

      GraphOptimizeState state1 = GraphOptimizeState{
          pcg1,
          0_ms,
      };
      GraphOptimizeState state2 = GraphOptimizeState{
          pcg2,
          0_ms,
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

      GraphOptimizeState state1 = GraphOptimizeState{
          pcg1,
          0_ms,
      };

      GraphOptimizeState state_ = GraphOptimizeState{
          other_pcg,
          0_ms,
      };

      CHECK_FALSE_WITHOUT_STRINGIFY(state1 == state_);
    }
  }

  TEST_CASE("GraphOptimizeState::operator<") {
    ParallelComputationGraph pcg1 = empty_parallel_computation_graph();
    ParallelComputationGraph pcg2 = empty_parallel_computation_graph();
    GraphOptimizeState state1 = GraphOptimizeState{
        pcg1,
        1.0_ms,
    };
    GraphOptimizeState state2 = GraphOptimizeState{
        pcg2,
        2.0_ms,
    };
    CHECK(state1 < state2);
  }
}
