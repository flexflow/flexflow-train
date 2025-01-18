#include "compiler/cost_estimator/task_simulator.h"
#include "compiler/cost_estimator/cost_estimator.h"
#include "compiler/cost_estimator/op_cost_estimate_key.h"
#include "compiler/cost_estimator/task_graph.dtg.h"
#include "compiler/cost_estimator/task_graph_traversal.h"
#include "compiler/cost_estimator/tasks_state_tracker.dtg.h"
#include "compiler/cost_estimator/timed_task.dtg.h"
#include "compiler/machine_mapping/device_mapping.dtg.h"
#include "compiler/machine_mapping/device_mapping.h"
#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/device_id.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "utils/containers/all_of.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/set_union.h"
#include "utils/containers/values.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/hash/unordered_multiset.h"
#include "utils/hash/unordered_set.h"
#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace FlexFlow {

static float
    single_parallel_layer_cost_estimator(parallel_layer_guid_t const &layer,
                                         ParallelComputationGraph const &pcg,
                                         MachineMapping const &machine_mapping,
                                         CostEstimator const &estimator) {
  MachineView mv = machine_mapping.machine_views.at(layer);
  return estimator.estimate_cost(
      get_mapped_op_cost_estimate_key_for_layer(pcg, layer, mv));
}

static float single_dependency_cost_estimator(
    ParallelComputationGraphEdge const &dependency,
    ParallelComputationGraph const &pcg,
    MachineMapping const &machine_mapping,
    CostEstimator const &estimator) {
  parallel_layer_guid_t incoming = get_src_layer(dependency);
  parallel_layer_guid_t outgoing = get_dst_layer(dependency);
  MachineView src_mv = machine_mapping.machine_views.at(incoming);
  MachineView dst_mv = machine_mapping.machine_views.at(outgoing);
  ParallelTensorShape tensor_shape = get_parallel_tensor_shape(
      pcg, parallel_tensor_guid_t{dependency.raw_edge.src});
  TensorSetMovement movement = TensorSetMovement{
      {SingleTensorMovement{tensor_shape, {src_mv}, {dst_mv}}}};
  return estimator.estimate_cost(movement);
}

static std::pair<
    DiGraph,
    bidict<Node,
           std::variant<parallel_layer_guid_t, ParallelComputationGraphEdge>>>
    get_digraph(ParallelComputationGraph const &pcg) {
  DiGraph digraph = DiGraph::create<AdjacencyDiGraph>();
  bidict<Node,
         std::variant<parallel_layer_guid_t, ParallelComputationGraphEdge>>
      node_map;

  for (parallel_layer_guid_t const &layer : get_parallel_layers(pcg)) {
    node_map.equate(digraph.add_node(), layer);
  }

  for (ParallelComputationGraphEdge const &edge : get_edges(pcg)) {
    node_map.equate(digraph.add_node(), edge);
  }

  for (auto const &[node, component] : node_map.as_unordered_map()) {
    if (std::holds_alternative<ParallelComputationGraphEdge>(component)) {
      auto edge = std::get<ParallelComputationGraphEdge>(component);
      parallel_layer_guid_t src_layer = get_src_layer(edge);
      parallel_layer_guid_t dst_layer = get_dst_layer(edge);

      Node src_node = node_map.at_r(src_layer);
      Node dst_node = node_map.at_r(dst_layer);

      digraph.add_edge(DirectedEdge{src_node, node});
      digraph.add_edge(DirectedEdge{node, dst_node});
    }
  }
  return {digraph, node_map};
}

static std::unordered_map<Node, float> get_cost_map(
    bidict<Node,
           std::variant<parallel_layer_guid_t,
                        ParallelComputationGraphEdge>> const &node_map,
    ParallelComputationGraph const &pcg,
    MachineMapping const &machine_mapping,
    CostEstimator const &estimator) {
  std::unordered_map<Node, float> cost_map;
  for (auto const &[node, component] : node_map) {
    if (std::holds_alternative<parallel_layer_guid_t>(component)) {
      parallel_layer_guid_t layer = std::get<parallel_layer_guid_t>(component);
      cost_map[node] = single_parallel_layer_cost_estimator(
          layer, pcg, machine_mapping, estimator);
    } else if (std::holds_alternative<ParallelComputationGraphEdge>(
                   component)) {
      ParallelComputationGraphEdge edge =
          std::get<ParallelComputationGraphEdge>(component);
      cost_map[node] = single_dependency_cost_estimator(
          edge, pcg, machine_mapping, estimator);
    }
  }
  return cost_map;
}

static bool is_allowed_to_run_super(
    Node const &task,
    TasksStateTracker const &state_tracker,
    DeviceMapping const &device_map,
    bidict<Node,
           std::variant<parallel_layer_guid_t,
                        ParallelComputationGraphEdge>> const &node_map) {
  auto component = node_map.at_l(task);

  if (std::holds_alternative<ParallelComputationGraphEdge>(component)) {
    return true;
  }

  parallel_layer_guid_t current_layer =
      std::get<parallel_layer_guid_t>(component);
  std::unordered_set<device_id_t> devices_occupied;

  for (TimedTask const &timed_task :
       state_tracker.tasks_processing.contents()) {
    auto task_component = node_map.at_l(timed_task.node);
    if (std::holds_alternative<parallel_layer_guid_t>(task_component)) {
      parallel_layer_guid_t processing_layer =
          std::get<parallel_layer_guid_t>(task_component);
      devices_occupied = set_union(
          devices_occupied, device_map.raw_device_map.at(processing_layer));
    }
  }

  auto required_devices = device_map.raw_device_map.at(current_layer);
  return intersection(devices_occupied, required_devices).empty();
}

float task_simulator_estimate_forward_pass_time(
    ParallelComputationGraph const &pcg,
    CostEstimator const &estimator,
    MachineMapping const &machine_mapping,
    MachineSpecification const &machine_spec) {

  DeviceMapping device_map =
      get_device_mapping(machine_mapping, machine_spec, pcg);

  auto [digraph, node_map] = get_digraph(pcg);
  auto cost_map = get_cost_map(node_map, pcg, machine_mapping, estimator);
  auto is_allowed_to_run = [&](Node const &task,
                               TasksStateTracker const &state_tracker) {
    return is_allowed_to_run_super(task, state_tracker, device_map, node_map);
  };
  TaskGraph task_graph = TaskGraph{digraph, cost_map, is_allowed_to_run};

  return simulate_forward_pass(task_graph);
}
} // namespace FlexFlow
