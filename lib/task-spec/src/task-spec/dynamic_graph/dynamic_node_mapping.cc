#include "task-spec/dynamic_graph/dynamic_node_mapping.h"
#include "utils/bidict/algorithms/bidict_transform_keys.h"
#include "utils/bidict/algorithms/bidict_transform_values.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

bidict<global_device_id_t, OperatorAtomicTaskShardBinding>
    dynamic_node_mapping_get_shard_bindings(DynamicNodeMapping const &m) {
  return bidict_transform_keys(
      m.op_task_group.get_shard_bindings(),
      [&](MachineSpaceCoordinate const &mc) -> global_device_id_t {
        return global_device_id_t{
            /*coord=*/mc,
            /*device_type=*/m.device_type,
        };
      });
}

OperatorAtomicTaskShardBinding
    dynamic_node_mapping_get_shard_binding_for_device(
        DynamicNodeMapping const &mapping,
        global_device_id_t const &device_id) {
  ASSERT(device_id.device_type == mapping.device_type);

  return mapping.op_task_group.get_shard_bindings().at_l(device_id.coord);
}

bidict<ParallelTensorSpaceCoordinate, global_device_id_t>
    dynamic_node_mapping_bindings_for_slot_name(
        DynamicNodeMapping const &mapping, TensorSlotName const &slot_name) {
  bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> coord_bindings =
      get_tensor_bindings_for_slot_name(mapping.op_task_group, slot_name);

  return bidict_transform_values(
      coord_bindings,
      [&](MachineSpaceCoordinate const &coord) -> global_device_id_t {
        return global_device_id_t{coord, mapping.device_type};
      });
}

std::set<global_device_id_t>
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
