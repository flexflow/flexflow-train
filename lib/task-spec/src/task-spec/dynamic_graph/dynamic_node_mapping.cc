#include "task-spec/dynamic_graph/dynamic_node_mapping.h"
#include "utils/bidict/algorithms/transform_values.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

bidict<ParallelTensorSpaceCoordinate, global_device_id_t>
    dynamic_node_mapping_bindings_for_slot_name(
        DynamicNodeMapping const &mapping, TensorSlotName const &slot_name) {
  bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> coord_bindings =
      get_tensor_bindings_for_slot_name(mapping.op_task_group, slot_name);

  return transform_values(
      coord_bindings,
      [&](MachineSpaceCoordinate const &coord) -> global_device_id_t {
        return global_device_id_t{coord, mapping.device_type};
      });
}

std::unordered_set<global_device_id_t>
    target_devices_of_dynamic_node_mapping(DynamicNodeMapping const &mapping) {

  return transform(mapping.op_task_group.get_shard_bindings().left_values(),
                   [&](MachineSpaceCoordinate const &c) -> global_device_id_t {
                     return global_device_id_t{
                         /*coord=*/c,
                         /*device_type=*/mapping.device_type,
                     };
                   });
}

} // namespace FlexFlow
