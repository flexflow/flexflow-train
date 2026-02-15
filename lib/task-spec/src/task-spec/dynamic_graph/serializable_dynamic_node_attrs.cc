#include "task-spec/dynamic_graph/serializable_dynamic_node_attrs.h"
#include <optional>

namespace FlexFlow {

SerializableDynamicNodeAttrs
    dynamic_node_attrs_to_serializable(DynamicNodeAttrs const &attrs) {
  return SerializableDynamicNodeAttrs{
      /*task_type=*/attrs.task_type,
      /*device_coord=*/attrs.device_coord,
      /*mapping=*/attrs.mapping,
      /*op_attrs=*/attrs.op_attrs,
      /*layer_guid=*/attrs.layer_guid,
  };
}

DynamicNodeAttrs dynamic_node_attrs_from_serializable(
    SerializableDynamicNodeAttrs const &attrs) {
  return DynamicNodeAttrs{
      /*task_type=*/attrs.task_type,
      /*device_coord=*/attrs.device_coord,
      /*mapping=*/attrs.mapping,
      /*op_attrs=*/attrs.op_attrs,
      /*layer_guid=*/attrs.layer_guid,
      /*per_device_op_state=*/std::nullopt,
  };
}

} // namespace FlexFlow
