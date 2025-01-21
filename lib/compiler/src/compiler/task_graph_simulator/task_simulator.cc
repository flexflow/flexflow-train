#include "compiler/task_graph_simulator/task_simulator.h"
#include "compiler/cost_estimator/cost_estimator.h"
#include "compiler/cost_estimator/op_cost_estimate_key.h"
#include "compiler/machine_mapping/unstructured_device_mapping.dtg.h"
#include "compiler/machine_mapping/unstructured_device_mapping.h"
#include "compiler/task_graph_simulator/pcg_task.dtg.h"
#include "compiler/task_graph_simulator/pcg_task_graph.h"
#include "compiler/task_graph_simulator/simulate_task_graph_execution.h"
#include "compiler/task_graph_simulator/task_execution_constraint.dtg.h"
#include "compiler/task_graph_simulator/task_graph_execution_trace.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/hash/unordered_set.h"
#include <unordered_set>

namespace FlexFlow {

float task_simulator_estimate_forward_pass_time(
    ParallelComputationGraph const &pcg,
    CostEstimator const &estimator,
    MachineMapping const &machine_mapping,
    MachineSpecification const &machine_spec) {

  PCGTaskGraph task_graph =
      get_pcg_task_graph(pcg, machine_mapping, machine_spec);

  auto cost_function = [&](Node const &node) -> float {
    PCGTask task = task_graph.node_to_task.at_l(node);
    if (task.is_operator()) {
      return estimator.estimate_cost(task.require_operator()).forward_runtime;
    } else {
      return estimator.estimate_cost(task.require_tensor_movement());
    }
  };

  auto is_allowed_to_run =
      [&](Node const &task,
          std::unordered_set<Node> const &in_progress_tasks,
          std::unordered_set<Node> const &finished_tasks) -> bool {
    PCGTask current_task = task_graph.node_to_task.at_l(task);

    UnstructuredDeviceMapping device_map =
        get_unstructured_device_mapping(machine_mapping, machine_spec, pcg);

    if (current_task.is_tensor_movement()) {
      return true;
    }
    assert(current_task.is_operator());

    auto get_devices = [&](Node const &n) {
      return task_graph.node_to_devices.at(n);
    };

    std::unordered_set<device_id_t> devices_occupied =
        set_union(transform(in_progress_tasks, get_devices));
    std::unordered_set<device_id_t> required_devices = get_devices(task);
    return intersection(devices_occupied, required_devices).empty();
  };

  TaskExecutionConstraint constraint =
      TaskExecutionConstraint{is_allowed_to_run};

  return get_total_execution_time(simulate_task_graph_execution(
      task_graph.graph, cost_function, constraint));
}

} // namespace FlexFlow
