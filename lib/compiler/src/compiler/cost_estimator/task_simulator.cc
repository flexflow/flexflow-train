#include "compiler/cost_estimator/task_simulator.h"
#include "compiler/cost_estimator/cost_estimator.h"
#include "compiler/cost_estimator/op_cost_estimate_key.h"
#include "compiler/cost_estimator/tasks_state_tracker.h"
#include "compiler/machine_mapping/device_mapping.h"
#include "pcg/device_id.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "utils/containers/all_of.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/values.h"
#include "utils/hash/unordered_multiset.h"
#include "utils/hash/unordered_set.h"
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

  TasksStateTracker state_tracker = [&]() {
    auto layer_cost_estimator = [&](parallel_layer_guid_t const &layer) {
      return single_parallel_layer_cost_estimator(
          layer, pcg, estimator, machine_mapping.machine_views.at(layer));
    };
    auto dependency_cost_estimator =
        [&](ParallelComputationGraphEdge const &dep) {
          return single_dependency_cost_estimator(
              dep, pcg, machine_mapping, estimator);
        };
    std::unordered_set<parallel_layer_guid_t> starting_layers =
        get_initial_layers(pcg);
    return TasksStateTracker{starting_layers,
                             {},
                             {},
                             0,
                             dependency_cost_estimator,
                             layer_cost_estimator,
                             pcg};
  }();

  DeviceMapping device_mapping =
      get_device_mapping(machine_mapping, machine_spec, pcg);

  std::unordered_map<device_id_t, bool> devices =
      generate_map(set_union(values(device_mapping.raw_device_map)),
                   [](device_id_t const &d) { return false; });

  auto layer_can_be_processed = [&](parallel_layer_guid_t const &layer) {
    auto layer_devices = device_mapping.raw_device_map.at(layer);
    return (all_of(layer_devices,
                   [&](device_id_t d) { return devices.at(d) == false; }));
  };

  auto occupy_devices = [&](parallel_layer_guid_t const &layer) {
    for (device_id_t d : device_mapping.raw_device_map.at(layer)) {
      devices.at(d) = true;
    }
  };

  auto free_devices = [&](parallel_layer_guid_t const &layer) {
    for (device_id_t d : device_mapping.raw_device_map.at(layer)) {
      devices.at(d) = false;
    }
  };

  while (!is_processing_done(state_tracker)) {
    auto frontier_copy = state_tracker.ready_layers;
    for (parallel_layer_guid_t const &layer : frontier_copy) {
      if (layer_can_be_processed(layer)) {
        start_layer_processing(state_tracker, layer);
        occupy_devices(layer);
      }
    }
    TimedComponent component = finish_processing_next_component(state_tracker);
    if (component.has<TimedLayer>()) {
      parallel_layer_guid_t layer = component.get<TimedLayer>().layer;
      free_devices(layer);
    }
  }

  return state_tracker.current_time;
}

} // namespace FlexFlow
