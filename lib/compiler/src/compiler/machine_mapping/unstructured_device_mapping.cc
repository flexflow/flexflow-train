#include "compiler/machine_mapping/unstructured_device_mapping.h"
#include "compiler/machine_mapping/machine_view.h"
#include "compiler/machine_mapping/unstructured_device_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_task_space.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/keys.h"
#include "utils/containers/map_values.h"

namespace FlexFlow {

UnstructuredDeviceMapping get_unstructured_device_mapping(
    MachineMapping const &machine_mapping,
    MachineComputeSpecification const &machine_spec,
    ParallelComputationGraph const &pcg) {
  std::unordered_map<parallel_layer_guid_t, std::unordered_set<device_id_t>>
      device_mapping;
  for (auto const &[layer, machine_view] : machine_mapping.machine_views) {
    OperatorTaskSpace op = get_operator_task_space(pcg, layer);
    device_mapping.insert(
        {layer, get_device_ids(op, machine_view, machine_spec)});
  }
  return UnstructuredDeviceMapping{device_mapping};
}

} // namespace FlexFlow
