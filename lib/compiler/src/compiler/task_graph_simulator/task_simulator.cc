#include "compiler/task_graph_simulator/task_simulator.h"
#include "compiler/cost_estimator/cost_estimator.h"
#include "compiler/cost_estimator/op_cost_estimate_key.h"
#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "compiler/machine_mapping/unstructured_device_mapping.dtg.h"
#include "compiler/machine_mapping/unstructured_device_mapping.h"
#include "compiler/task_graph_simulator/in_progress_task.dtg.h"
#include "compiler/task_graph_simulator/pcg_task.dtg.h"
#include "compiler/task_graph_simulator/pcg_task_graph.dtg.h"
#include "compiler/task_graph_simulator/pcg_task_graph.h"
#include "compiler/task_graph_simulator/simulate_task_graph_execution.h"
#include "compiler/task_graph_simulator/task_execution_constraint.dtg.h"
#include "compiler/task_graph_simulator/task_graph_execution_state.dtg.h"
#include "compiler/task_graph_simulator/task_graph_execution_trace.h"
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
#include "utils/containers/filter.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
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

float task_simulator_estimate_forward_pass_time(
    ParallelComputationGraph const &pcg,
    CostEstimator const &estimator,
    MachineMapping const &machine_mapping,
    MachineSpecification const &machine_spec) {

  UnstructuredDeviceMapping device_map =
      get_unstructured_device_mapping(machine_mapping, machine_spec, pcg);

  PCGTaskGraph task_graph = get_pcg_task_graph(pcg);

  auto cost_function = [&](Node const &node) {
    PCGTask task = task_graph.node_map.at_l(node);
    if (task.is_layer()) {
      return single_parallel_layer_cost_estimator(
          task.require_layer(), pcg, machine_mapping, estimator);
    } else {
      return single_dependency_cost_estimator(
          task.require_edge(), pcg, machine_mapping, estimator);
    }
  };

  auto is_allowed_to_run =
      [&](Node const &task,
          std::unordered_set<Node> const &in_progress_tasks,
          std::unordered_set<Node> const &finished_tasks) -> bool {
    PCGTask current_task = task_graph.node_map.at_l(task);

    if (current_task.is_edge()) {
      return true;
    }

    auto layers = filtrans(in_progress_tasks, [&](Node const &n) {
      PCGTask task = task_graph.node_map.at_l(n);
      return task.is_layer() ? std::make_optional(task.require_layer())
                             : std::nullopt;
    });

    std::unordered_set<device_id_t> devices_occupied =
        set_union(transform(layers, [&](parallel_layer_guid_t const &layer) {
          return device_map.raw_device_map.at(layer);
        }));

    std::unordered_set<device_id_t> required_devices =
        device_map.raw_device_map.at(current_task.require_layer());

    return intersection(devices_occupied, required_devices).empty();
  };

  TaskExecutionConstraint constraint =
      TaskExecutionConstraint{is_allowed_to_run};

  return get_endtime(simulate_task_graph_execution(
      task_graph.graph, cost_function, constraint));
}

} // namespace FlexFlow
