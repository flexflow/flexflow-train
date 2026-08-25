#include "task-spec/dynamic_graph/dynamic_node_mapping.h"
#include "task-spec/dynamic_graph/parallel_tensor_mapping.h"
#include "utils/bidict/algorithms/bidict_transform_keys.h"
#include "utils/bidict/algorithms/bidict_transform_values.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/require_all_same.h"
#include "utils/containers/transform.h"
#include "utils/optional.h"

namespace FlexFlow {

DynamicNodeMapping dynamic_node_mapping_from_value_mapping(
    std::map<DynamicTensorSlot, ParallelTensorMapping> const &mapping) {
  MappedOperatorTaskGroup op_task_group;
  DeviceType device_type = assert_unwrap(require_all_same(
      transform(flatmap(values(mapping),
                        [](ParallelTensorMapping const &m) {
                          return pt_mapping_get_device_set(m);
                        }),
                [](global_device_id_t const &id) { return id.device_type; })));
  return DynamicNodeMapping{
      /*op_task_group=*/op_task_group,
      /*device_type=*/device_type,
  };
}

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
