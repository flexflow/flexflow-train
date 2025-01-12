#include "compiler/cost_estimator/task_simulator.h"
#include "compiler/cost_estimator/cost_estimator.h"
#include "compiler/cost_estimator/op_cost_estimate_key.h"
#include "compiler/cost_estimator/single_tensor_movement.dtg.h"
#include "compiler/cost_estimator/tasks_state_tracker.dtg.h"
#include "compiler/cost_estimator/tensor_set_movement.dtg.h"
#include "compiler/cost_estimator/timed_component.dtg.h"
#include "compiler/cost_estimator/timed_dependency.dtg.h"
#include "compiler/cost_estimator/timed_layer.dtg.h"
#include "compiler/machine_mapping/device_mapping.h"
#include "compiler/machine_mapping/machine_mapping.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/device_id.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "utils/containers/all_of.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_one_of.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/keys.h"
#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/values.h"
#include "utils/graph/dataflow_graph/algorithms/get_outgoing_edges.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_graph_inputs.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_source_nodes.h"
#include "utils/hash/unordered_set.h"
#include "utils/overload.h"
#include <optional>
#include <unordered_set>

namespace FlexFlow {

static float
    single_parallel_layer_cost_estimator(parallel_layer_guid_t const &layer,
                                         ParallelComputationGraph const &pcg,
                                         CostEstimator const &estimator,
                                         MachineView const &mv) {
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

float task_simulator_estimate_forward_pass_time(
    ParallelComputationGraph const &pcg,
    CostEstimator const &estimator,
    MachineMapping const &machine_mapping,
    MachineSpecification const &machine_spec) {

  float current_time = 0.0f;

  TasksStateTracker state_tracker =
      TasksStateTracker(get_initial_layers(pcg), {}, {});

  DeviceMapping device_mapping =
      get_device_mapping(machine_mapping, machine_spec, pcg);

  std::unordered_map<device_id_t, bool> devices =
      generate_map(set_union(values(device_mapping.raw_device_map)),
                   [](device_id_t const &d) { return false; });

  auto start_layer_processing = [&](parallel_layer_guid_t const &layer) {
    float cost = single_parallel_layer_cost_estimator(
        layer, pcg, estimator, machine_mapping.machine_views.at(layer));
    state_tracker.component_processing.push(
        TimedComponent{TimedLayer{current_time + cost, layer}});
    for (device_id_t d : device_mapping.raw_device_map.at(layer)) {
      devices.at(d) = true;
    }
    state_tracker.ready_layers.erase(layer);
  };

  auto start_dependency_processing =
      [&](ParallelComputationGraphEdge const &dependency, float start_time) {
        float cost = single_dependency_cost_estimator(
            dependency, pcg, machine_mapping, estimator);
        state_tracker.component_processing.push(
            TimedComponent{TimedDependency{start_time + cost, dependency}});
      };

  auto finish_layer_processing = [&](TimedLayer const &timed_layer) {
    for (device_id_t d : device_mapping.raw_device_map.at(timed_layer.layer)) {
      devices.at(d) = false;
    }
    state_tracker.processed_components.insert(TimedComponent{timed_layer});
    current_time = timed_layer.endtime;
    std::unordered_set<ParallelComputationGraphEdge> outgoing_dependencies =
        get_outgoing_edges(pcg, timed_layer.layer);
    for (ParallelComputationGraphEdge const &dep : outgoing_dependencies) {
      start_dependency_processing(dep, timed_layer.endtime);
    }
  };

  auto finish_dependency_processing =
      [&](TimedDependency const &timed_dependency) {
        state_tracker.processed_components.insert(
            TimedComponent{timed_dependency});
        parallel_layer_guid_t destination_layer =
            get_dst_layer(timed_dependency.raw_edge);
        std::unordered_set<ParallelComputationGraphEdge> incoming_dependencies =
            get_incoming_edges(pcg, destination_layer);
        std::unordered_set<ParallelComputationGraphEdge>
            non_timed_processed_dependencies = filtrans(
                state_tracker.processed_components,
                [](TimedComponent const &component) {
                  return component
                      .visit<std::optional<ParallelComputationGraphEdge>>(
                          overload{[&](TimedLayer const &layer) {
                                     return std::nullopt;
                                   },
                                   [&](TimedDependency const &dep) {
                                     return dep.raw_edge;
                                   }});
                });
        // start processing a new node if all dependencies have been processed
        // already
        if (is_subseteq_of(incoming_dependencies,
                           non_timed_processed_dependencies)) {
          state_tracker.ready_layers.insert(destination_layer);
        }
        current_time = timed_dependency.endtime;
      };

  while (!state_tracker.ready_layers.empty() ||
         !state_tracker.component_processing.empty()) {

    auto frontier_copy = state_tracker.ready_layers;
    for (parallel_layer_guid_t const &layer : frontier_copy) {
      auto layer_devices = device_mapping.raw_device_map.at(layer);
      if (all_of(layer_devices,
                 [&](device_id_t d) { return devices.at(d) == false; })) {
        start_layer_processing(layer);
      }
    }

    if (!state_tracker.component_processing.empty()) {
      TimedComponent component = state_tracker.component_processing.top();
      state_tracker.component_processing.pop();

      if (component.has<TimedDependency>()) {
        finish_dependency_processing(component.get<TimedDependency>());
      } else if (component.has<TimedLayer>()) {
        finish_layer_processing(component.get<TimedLayer>());
      }
    }
  }

  return current_time;
}

} // namespace FlexFlow
