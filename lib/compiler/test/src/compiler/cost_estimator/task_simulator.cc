#include "compiler/cost_estimator/task_simulator.h"
#include "../cost_estimator_for_test.h"
#include "compiler/cost_estimator/cost_estimator.h"
#include "compiler/cost_estimator/timed_layer.dtg.h"
#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
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
#include <doctest/doctest.h>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

TEST_SUITE(FF_TEST_SUITE) {
  CostEstimator estimator =
      make_fake_constant_cost_estimator(/*op_cost*/ 10.0f, /*comm_cost*/ 1.0f);
  MachineSpecification machine_spec = MachineSpecification{3, 3, 3, 1, 1};

  TEST_CASE("task_simulator: linear graph") {
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

    parallel_layer_guid_t layer0 = get_source_layer(tensor0);
    parallel_layer_guid_t layer1 = get_source_layer(tensor1);

    ParallelComputationGraph pcg = b.pcg;

    std::unordered_set<parallel_layer_guid_t> layers = {layer0, layer1};
    CHECK(get_parallel_layers(pcg) == layers);

    MachineView mv1 = MachineView{
        MachineSpaceCoordinate{0, 0, DeviceType::GPU},
        {
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
        }};
    MachineView mv2 = MachineView{
        MachineSpaceCoordinate{0, 1, DeviceType::GPU},
        {
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
        }};

    MachineMapping device_mapping = MachineMapping{{
        {layer0, mv1},
        {layer1, mv2},
    }};

    float result = task_simulator_forward_pass(
        pcg, estimator, device_mapping, machine_spec);
    float correct = 10 + 1 + 10;
    CHECK(result == correct);
  }

  TEST_CASE("task_simulator: rhombus graph, all separate devices") {
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

    std::unordered_set<parallel_layer_guid_t> layers = {
        layer0, layer1, layer2, layer3};
    CHECK(get_parallel_layers(pcg) == layers);

    MachineView mv0 = MachineView{
        MachineSpaceCoordinate{0, 0, DeviceType::GPU},
        {
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
        }};
    MachineView mv1 = MachineView{
        MachineSpaceCoordinate{0, 1, DeviceType::GPU},
        {
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
        }};
    MachineView mv2 = MachineView{
        MachineSpaceCoordinate{1, 0, DeviceType::GPU},
        {
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
        }};
    MachineView mv3 = MachineView{
        MachineSpaceCoordinate{1, 1, DeviceType::GPU},
        {
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
        }};

    MachineMapping device_mapping = MachineMapping{{
        {layer0, mv0},
        {layer1, mv1},
        {layer2, mv2},
        {layer3, mv3},
    }};

    float result = task_simulator_forward_pass(
        pcg, estimator, device_mapping, machine_spec);
    float correct = 10 + 1 + 10 + 1 + 10;
    CHECK(result == correct);
  }

  TEST_CASE("task_simulator: rhombus graph, all same device") {
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

    std::unordered_set<parallel_layer_guid_t> layers = {
        layer0, layer1, layer2, layer3};
    CHECK(get_parallel_layers(pcg) == layers);

    MachineView mv = MachineView{
        MachineSpaceCoordinate{0, 0, DeviceType::GPU},
        {
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
            MachineViewDimension{stride_t{1},
                                 MachineSpecificationDimension::INTER_NODE},
        }};
    MachineMapping device_mapping = MachineMapping{{
        {layer0, mv},
        {layer1, mv},
        {layer2, mv},
        {layer3, mv},
    }};

    float result = task_simulator_forward_pass(
        pcg, estimator, device_mapping, machine_spec);
    float correct = 10 + 10 + 10 + 10 + 1 + 1;
    CHECK(result == correct);
  }
}
} // namespace FlexFlow
