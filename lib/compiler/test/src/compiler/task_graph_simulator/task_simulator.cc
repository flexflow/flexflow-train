#include "compiler/task_graph_simulator/task_simulator.h"
#include "compiler/cost_estimator/cost_estimator.h"
#include "compiler/cost_estimator/op_cost_metrics.dtg.h"
#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
#include "internal/runtime_only_cost_estimator_for_test.h"
#include "op-attrs/ops/input_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dims.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/device_id.h"
#include "pcg/device_type.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_specification_dimension.dtg.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/machine_view.h"
#include "pcg/machine_view_dimension.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.h"
#include "pcg/stride_t.dtg.h"
#include "substitutions/sub_parallel_computation_graph.dtg.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "utils/containers/get_only.h"
#include "utils/deduplicated_priority_queue.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_source_nodes.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <doctest/doctest.h>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("task_simulator_estimate_forward_pass_time") {
    MachineSpecification machine_spec =
        MachineSpecification{/*num_nodes=*/3_p,
                             /*num_cpus_per_node=*/3_p,
                             /*num_gpus_per_node=*/3_p,
                             /*inter_node_bandwidth=*/1.0f,
                             /*intra_node_bandwidth=*/1.0f};

    SUBCASE("linear graph") {
      ParallelComputationGraphBuilder b;
      TensorShape input_shape = TensorShape{
          TensorDims{
              FFOrdered{
                  10_p,
                  7_p,
              },
          },
          DataType::FLOAT,
      };
      parallel_tensor_guid_t tensor0 = b.create_input_tensor(input_shape);
      parallel_tensor_guid_t tensor1 = b.relu(tensor0);

      parallel_layer_guid_t layer0 = get_source_layer(tensor0);
      parallel_layer_guid_t layer1 = get_source_layer(tensor1);

      std::vector<MachineViewDimension> dims = {
          MachineViewDimension{stride_t{1_p},
                               MachineSpecificationDimension::INTER_NODE},
          MachineViewDimension{stride_t{1_p},
                               MachineSpecificationDimension::INTER_NODE},
          MachineViewDimension{stride_t{1_p},
                               MachineSpecificationDimension::INTER_NODE},
          MachineViewDimension{stride_t{1_p},
                               MachineSpecificationDimension::INTER_NODE},
      };
      ParallelComputationGraph pcg = b.pcg;
      MachineView mv1 =
          MachineView{MachineSpaceCoordinate{0_n, 0_n, DeviceType::GPU}, dims};
      MachineView mv2 =
          MachineView{MachineSpaceCoordinate{0_n, 1_n, DeviceType::GPU}, dims};

      MachineMapping device_mapping = MachineMapping{{
          {layer0, mv1},
          {layer1, mv2},
      }};

      SUBCASE("constant op, comm cost") {
        RuntimeOnlyCostEstimator estimator =
            make_fake_constant_runtime_only_cost_estimator(
                /*forward_op_cost=*/10_ms,
                /*backward_op_cost=*/10_ms,
                /*comm_cost=*/1_ms);

        milliseconds_t result = task_simulator_estimate_forward_pass_time(
            pcg, estimator, device_mapping, machine_spec);

        milliseconds_t correct = 10_ms + 1_ms + 10_ms;
        CHECK(result == correct);
      }

      SUBCASE("variable op, comm cost") {
        RuntimeOnlyCostEstimator cost_estimator =
            make_fake_runtime_only_cost_estimator(
                [](RuntimeOnlyOpCostEstimateKey const &key) {
                  if (key.op_attrs.has<InputAttrs>()) {
                    return RuntimeOnlyOpCostMetrics{
                        /*forward_runtime=*/10_ms,
                        /*backward_runtime=*/10_ms,
                    }; // layer0
                  } else if (key.op_attrs.has<ElementUnaryAttrs>()) {
                    return RuntimeOnlyOpCostMetrics{
                        /*forward_runtime=*/1_ms,
                        /*backward_runtime=*/1_ms,
                    }; // layer1
                  } else {
                    return RuntimeOnlyOpCostMetrics{
                        /*forward_runtime=*/0_ms,
                        /*backward_runtime=*/0_ms,
                    };
                  }
                },
                [](TensorSetMovement const &comm) { return 5_ms; });

        milliseconds_t result = task_simulator_estimate_forward_pass_time(
            pcg, cost_estimator, device_mapping, machine_spec);
        milliseconds_t correct = 10_ms + 5_ms + 1_ms;
        CHECK(result == correct);
      }
    }

    SUBCASE("rhomboidal graph") {
      ParallelComputationGraphBuilder b;

      TensorShape input_shape = TensorShape{
          TensorDims{
              FFOrdered{
                  10_p,
                  1_p,
              },
          },
          DataType::FLOAT,
      };

      parallel_tensor_guid_t tensor0 = b.create_input_tensor(input_shape);
      parallel_tensor_guid_t tensor1 = b.relu(tensor0);
      parallel_tensor_guid_t tensor2 = b.relu(tensor0);
      parallel_tensor_guid_t tensor3 = b.add(tensor1, tensor2);

      parallel_layer_guid_t layer0 = get_source_layer(tensor0);
      parallel_layer_guid_t layer1 = get_source_layer(tensor1);
      parallel_layer_guid_t layer2 = get_source_layer(tensor2);
      parallel_layer_guid_t layer3 = get_source_layer(tensor3);

      ParallelComputationGraph pcg = b.pcg;
      std::vector<MachineViewDimension> dims = {
          MachineViewDimension{stride_t{1_p},
                               MachineSpecificationDimension::INTER_NODE},
          MachineViewDimension{stride_t{1_p},
                               MachineSpecificationDimension::INTER_NODE},
          MachineViewDimension{stride_t{1_p},
                               MachineSpecificationDimension::INTER_NODE},
          MachineViewDimension{stride_t{1_p},
                               MachineSpecificationDimension::INTER_NODE},
      };

      SUBCASE("all different devices") {
        MachineView mv0 = MachineView{
            MachineSpaceCoordinate{0_n, 0_n, DeviceType::GPU}, dims};
        MachineView mv1 = MachineView{
            MachineSpaceCoordinate{0_n, 1_n, DeviceType::GPU}, dims};
        MachineView mv2 = MachineView{
            MachineSpaceCoordinate{1_n, 0_n, DeviceType::GPU}, dims};
        MachineView mv3 = MachineView{
            MachineSpaceCoordinate{1_n, 1_n, DeviceType::GPU}, dims};

        MachineMapping device_mapping = MachineMapping{{
            {layer0, mv0},
            {layer1, mv1},
            {layer2, mv2},
            {layer3, mv3},
        }};

        SUBCASE("constant op, comm cost") {
          RuntimeOnlyCostEstimator estimator =
              make_fake_constant_runtime_only_cost_estimator(
                  /*forward_op_cost=*/10_ms,
                  /*backward_op_cost=*/10_ms,
                  /*comm_cost=*/1_ms);

          milliseconds_t result = task_simulator_estimate_forward_pass_time(
              pcg, estimator, device_mapping, machine_spec);
          milliseconds_t correct = 10_ms + 1_ms + 10_ms + 1_ms + 10_ms;
          CHECK(result == correct);
        }

        SUBCASE("variable op, comm cost") {
          RuntimeOnlyCostEstimator cost_estimator =
              make_fake_runtime_only_cost_estimator(
                  [](RuntimeOnlyOpCostEstimateKey const &key) {
                    if (key.op_attrs.has<InputAttrs>()) {
                      return RuntimeOnlyOpCostMetrics{
                          /*forward_runtime=*/10_ms,
                          /*backward_runtime=*/10_ms,
                      }; // layer0
                    } else if (key.op_attrs.has<ElementUnaryAttrs>()) {
                      return RuntimeOnlyOpCostMetrics{
                          /*forward_runtime=*/1_ms,
                          /*backward_runtime=*/1_ms,
                      }; // layers 1, 2
                    } else if (key.op_attrs.has<ElementBinaryAttrs>()) {
                      return RuntimeOnlyOpCostMetrics{
                          /*forward_runtime=*/2_ms,
                          /*backward_runtime=*/2_ms,
                      }; // layer3
                    } else {
                      return RuntimeOnlyOpCostMetrics{
                          /*forward_runtime=*/0_ms,
                          /*backward_runtime=*/0_ms,
                      };
                    }
                  },
                  [](TensorSetMovement const &comm) { return 5_ms; });
        }
      }

      SUBCASE("all the same device") {
        MachineView mv = MachineView{
            MachineSpaceCoordinate{0_n, 0_n, DeviceType::GPU}, dims};
        MachineMapping device_mapping = MachineMapping{{
            {layer0, mv},
            {layer1, mv},
            {layer2, mv},
            {layer3, mv},
        }};

        SUBCASE("constant op, cost cost") {
          RuntimeOnlyCostEstimator cost_estimator =
              make_fake_constant_runtime_only_cost_estimator(
                  /*forward_op_cost=*/10_ms,
                  /*backward_op_cost=*/10_ms,
                  /*comm_cost=*/1_ms);

          milliseconds_t result = task_simulator_estimate_forward_pass_time(
              pcg, cost_estimator, device_mapping, machine_spec);
          milliseconds_t correct = 10_ms + 10_ms + 10_ms + 10_ms + 1_ms + 1_ms;
          CHECK(result == correct);
        }

        SUBCASE("variable op, cost cost") {
          RuntimeOnlyCostEstimator cost_estimator =
              make_fake_runtime_only_cost_estimator(
                  [](RuntimeOnlyOpCostEstimateKey const &key) {
                    if (key.op_attrs.has<InputAttrs>()) {
                      return RuntimeOnlyOpCostMetrics{
                          /*forward_runtime=*/10_ms,
                          /*backward_runtime=*/10_ms,
                      }; // layer0
                    } else if (key.op_attrs.has<ElementUnaryAttrs>()) {
                      return RuntimeOnlyOpCostMetrics{
                          /*forward_runtime=*/1_ms,
                          /*backward_runtime=*/1_ms,
                      }; // layers 1, 2
                    } else if (key.op_attrs.has<ElementBinaryAttrs>()) {
                      return RuntimeOnlyOpCostMetrics{
                          /*forward_runtime=*/2_ms,
                          /*backward_runtime=*/2_ms,
                      }; // layer3
                    } else {
                      return RuntimeOnlyOpCostMetrics{
                          /*forward_runtime=*/0_ms,
                          /*backward_runtime=*/0_ms,
                      };
                    }
                  },
                  [](TensorSetMovement const &comm) { return 5_ms; });
          milliseconds_t result = task_simulator_estimate_forward_pass_time(
              pcg, cost_estimator, device_mapping, machine_spec);
          milliseconds_t correct = 10_ms + 5_ms + (1_ms + 1_ms) + 5_ms + 2_ms;
          CHECK(result == correct);
        }
      }
    }
  }
}
} // namespace FlexFlow
