#include "compiler/task_graph_simulator/task_simulator.h"
#include "../cost_estimator_for_test.h"
#include "compiler/cost_estimator/cost_estimator.h"
#include "compiler/cost_estimator/op_cost_metrics.dtg.h"
#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
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
        MachineSpecification{/*num_nodes=*/3,
                             /*num_cpus_per_node=*/3,
                             /*num_gpus_per_node=*/3,
                             /*inter_node_bandwidth=*/1.0f,
                             /*intra_node_bandwidth=*/1.0f};

    SUBCASE("linear graph") {
      ParallelComputationGraphBuilder b;
      ParallelTensorShape input_shape = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{},
              ReplicaParallelDimSet{
                  SumDegree{1},
                  DiscardCopyDegree{1},
              },
          },
          DataType::FLOAT,
      };
      parallel_tensor_guid_t tensor0 = b.create_input_tensor(input_shape);
      parallel_tensor_guid_t tensor1 = b.relu(tensor0);

      parallel_layer_guid_t layer0 = get_source_layer(tensor0);
      parallel_layer_guid_t layer1 = get_source_layer(tensor1);

      std::vector<MachineViewDimension> dims = {
          MachineViewDimension{stride_t{1},
                               MachineSpecificationDimension::INTER_NODE},
          MachineViewDimension{stride_t{1},
                               MachineSpecificationDimension::INTER_NODE},
      };
      ParallelComputationGraph pcg = b.pcg;
      MachineView mv1 =
          MachineView{MachineSpaceCoordinate{0, 0, DeviceType::GPU}, dims};
      MachineView mv2 =
          MachineView{MachineSpaceCoordinate{0, 1, DeviceType::GPU}, dims};

      MachineMapping device_mapping = MachineMapping{{
          {layer0, mv1},
          {layer1, mv2},
      }};

      SUBCASE("constant op, comm cost") {
        CostEstimator estimator = make_fake_constant_cost_estimator(
            /*forward_op_cost=*/10.0f,
            /*backward_op_cost=*/10.0f,
            /*comm_cost=*/1.0f,
            /*memory_cost=*/nonnegative_int{0});

        float result = task_simulator_estimate_forward_pass_time(
            pcg, estimator, device_mapping, machine_spec);

        float correct = 10 + 1 + 10;
        CHECK(result == correct);
      }

      SUBCASE("variable op, comm cost") {
        CostEstimator cost_estimator = make_fake_cost_estimator(
            [](OpCostEstimateKey const &op) {
              if (op.op_attrs.has<InputAttrs>()) {
                return OpCostMetrics{/*forward_runtime=*/10.0f,
                                     /*backward_runtime=*/10.0f,
                                     /*memory=*/nonnegative_int{0}}; // layer0
              }
              if (op.op_attrs.has<ElementUnaryAttrs>()) {
                return OpCostMetrics{/*forward_runtime=*/1.0f,
                                     /*backward_runtime=*/1.0f,
                                     /*memory=*/nonnegative_int{0}}; // layer1
              }
              return OpCostMetrics{/*forward_runtime=*/0.0f,
                                   /*backward_runtime=*/0.0f,
                                   /*memory=*/nonnegative_int{0}};
            },
            [](TensorSetMovement const &comm) { return 5.0f; });

        float result = task_simulator_estimate_forward_pass_time(
            pcg, cost_estimator, device_mapping, machine_spec);
        float correct = 10 + 5 + 1;
        CHECK(result == correct);
      }
    }

    SUBCASE("rhomboidal graph") {
      ParallelComputationGraphBuilder b;

      ParallelTensorShape input_shape = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{ShardParallelDim{10, 1}},
              ReplicaParallelDimSet{
                  SumDegree{1},
                  DiscardCopyDegree{1},
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
          MachineViewDimension{stride_t{1},
                               MachineSpecificationDimension::INTER_NODE},
          MachineViewDimension{stride_t{1},
                               MachineSpecificationDimension::INTER_NODE},
          MachineViewDimension{stride_t{1},
                               MachineSpecificationDimension::INTER_NODE},
      };

      SUBCASE("all different devices") {
        MachineView mv0 =
            MachineView{MachineSpaceCoordinate{0, 0, DeviceType::GPU}, dims};
        MachineView mv1 =
            MachineView{MachineSpaceCoordinate{0, 1, DeviceType::GPU}, dims};
        MachineView mv2 =
            MachineView{MachineSpaceCoordinate{1, 0, DeviceType::GPU}, dims};
        MachineView mv3 =
            MachineView{MachineSpaceCoordinate{1, 1, DeviceType::GPU}, dims};

        MachineMapping device_mapping = MachineMapping{{
            {layer0, mv0},
            {layer1, mv1},
            {layer2, mv2},
            {layer3, mv3},
        }};
        SUBCASE("constant op, comm cost") {
          CostEstimator estimator = make_fake_constant_cost_estimator(
              /*forward_op_cost=*/10.0f,
              /*backward_op_cost=*/10.0f,
              /*comm_cost=*/1.0f,
              /*memory_cost=*/nonnegative_int{0});

          float result = task_simulator_estimate_forward_pass_time(
              pcg, estimator, device_mapping, machine_spec);
          float correct = 10 + 1 + 10 + 1 + 10;
          CHECK(result == correct);
        }
        SUBCASE("variable op, comm cost") {
          CostEstimator cost_estimator = make_fake_cost_estimator(
              [](OpCostEstimateKey const &op) {
                if (op.op_attrs.has<InputAttrs>()) {
                  return OpCostMetrics{/*forward_runtime=*/10.0f,
                                       /*backward_runtime=*/10.0f,
                                       /*memory=*/nonnegative_int{0}}; // layer0
                }
                if (op.op_attrs.has<ElementUnaryAttrs>()) {
                  return OpCostMetrics{
                      /*forward_runtime=*/1.0f,
                      /*backward_runtime=*/1.0f,
                      /*memory=*/nonnegative_int{0}}; // layers 1, 2
                }
                if (op.op_attrs.has<ElementBinaryAttrs>()) {
                  return OpCostMetrics{/*forward_runtime=*/2.0f,
                                       /*backward_runtime=*/2.0f,
                                       /*memory=*/nonnegative_int{0}}; // layer3
                }
                return OpCostMetrics{/*forward_runtime=*/0.0f,
                                     /*backward_runtime=*/0.0f,
                                     /*memory=*/nonnegative_int{0}};
              },
              [](TensorSetMovement const &comm) { return 5.0f; });
        }
      }

      SUBCASE("all the same device") {
        MachineView mv =
            MachineView{MachineSpaceCoordinate{0, 0, DeviceType::GPU}, dims};
        MachineMapping device_mapping = MachineMapping{{
            {layer0, mv},
            {layer1, mv},
            {layer2, mv},
            {layer3, mv},
        }};
        SUBCASE("constant op, cost cost") {
          CostEstimator cost_estimator = make_fake_constant_cost_estimator(
              /*forward_op_cost=*/10.0f,
              /*backward_op_cost=*/10.0f,
              /*comm_cost=*/1.0f,
              /*memory_cost=*/nonnegative_int{0});

          float result = task_simulator_estimate_forward_pass_time(
              pcg, cost_estimator, device_mapping, machine_spec);
          float correct = 10 + 10 + 10 + 10 + 1 + 1;
          CHECK(result == correct);
        }
        SUBCASE("variable op, cost cost") {
          CostEstimator cost_estimator = make_fake_cost_estimator(
              [](OpCostEstimateKey const &op) {
                if (op.op_attrs.has<InputAttrs>()) {
                  return OpCostMetrics{/*forward_runtime=*/10.0f,
                                       /*backward_runtime=*/10.0f,
                                       /*memory=*/nonnegative_int{0}}; // layer0
                }
                if (op.op_attrs.has<ElementUnaryAttrs>()) {
                  return OpCostMetrics{
                      /*forward_runtime=*/1.0f,
                      /*backward_runtime=*/1.0f,
                      /*memory=*/nonnegative_int{0}}; // layers 1, 2
                }
                if (op.op_attrs.has<ElementBinaryAttrs>()) {
                  return OpCostMetrics{/*forward_runtime=*/2.0f,
                                       /*backward_runtime=*/2.0f,
                                       /*memory=*/nonnegative_int{0}}; // layer3
                }
                return OpCostMetrics{/*forward_runtime=*/0.0f,
                                     /*backward_runtime=*/0.0f,
                                     /*memory=*/nonnegative_int{0}};
              },
              [](TensorSetMovement const &comm) { return 5.0f; });
          float result = task_simulator_estimate_forward_pass_time(
              pcg, cost_estimator, device_mapping, machine_spec);
          float correct = 10 + 5 + (1 + 1) + 5 + 2;
          CHECK(result == correct);
        }
      }
    }
  }
}
} // namespace FlexFlow
