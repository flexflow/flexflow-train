#include "compiler/unity_algorithm/graph_optimize_state.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("GraphOptimizeState::operator== and operator!=") {
    ParallelComputationGraph pcg1 = empty_parallel_computation_graph();
    ParallelComputationGraph pcg2 = empty_parallel_computation_graph();

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{10, 1},
            },
            ReplicaParallelDimSet{
                SumDegree{1},
                DiscardCopyDegree{1},
            },
        },
        DataType::FLOAT,
    };

    auto make_output_attrs = [](ParallelTensorShape const &shape) {
      return ParallelTensorAttrs{
          /*shape=*/shape,
          /*sync_type=*/std::nullopt,
          /*initializer=*/std::nullopt,
          /*create_gradients=*/CreateGrad::YES,
      };
    };

    auto make_layer_attrs = [](PCGOperatorAttrs const &op_attrs) {
      return ParallelLayerAttrs{
          /*op_attrs=*/op_attrs,
          /*name=*/std::nullopt,
      };
    };

    PCGOperatorAttrs input_attrs = PCGOperatorAttrs{InputAttrs{}};

    add_parallel_layer(pcg2,
                       /*layer_attrs=*/make_layer_attrs(input_attrs),
                       /*inputs=*/{},
                       /*output_labels=*/{make_output_attrs(input_shape)});

    SUBCASE("same pcgs") {
      GraphOptimizeState state1 = GraphOptimizeState(pcg1, 0.0);
      GraphOptimizeState state2 = GraphOptimizeState(pcg1, 0.0);
      bool result_eq = state1 == state2;
      bool expected_eq = true;
      CHECK(result_eq == expected_eq);
      bool result_neq = state1 != state2;
      bool expected_neq = false;
      CHECK(result_neq == expected_neq);
    }

    SUBCASE("different pcgs with the same runtime") {
      GraphOptimizeState state1 = GraphOptimizeState(pcg1, 1.0);
      GraphOptimizeState state2 = GraphOptimizeState(pcg2, 1.0);
      bool result_eq = state1 == state2;
      bool expected_eq = false;
      CHECK(result_eq == expected_eq);
      bool result_neq = state1 != state2;
      bool expected_neq = true;
      CHECK(result_neq == expected_neq);
    }

    SUBCASE("different pcgs with different runtime") {
      GraphOptimizeState state1 = GraphOptimizeState(pcg1, 1.0);
      GraphOptimizeState state2 = GraphOptimizeState(pcg2, 2.0);
      bool result_eq = state1 == state2;
      bool expected_eq = false;
      CHECK(result_eq == expected_eq);
      bool result_neq = state1 != state2;
      bool expected_neq = true;
      CHECK(result_neq == expected_neq);
    }
  }

  TEST_CASE("GraphOptimizeState::operator<") {
    ParallelComputationGraph pcg1 = empty_parallel_computation_graph();
    ParallelComputationGraph pcg2 = empty_parallel_computation_graph();
    GraphOptimizeState state1 = GraphOptimizeState(pcg1, 1.0);
    GraphOptimizeState state2 = GraphOptimizeState(pcg2, 2.0);
    bool result = state1 < state2;
    bool expected = true;
    CHECK(result == expected);
  }
}
