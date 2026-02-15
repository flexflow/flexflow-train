#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_attrs.h"
#include "task-spec/dynamic_graph/serializable_dynamic_value_attrs.h"
#include "utils/containers/map_values.h"

namespace FlexFlow {

SerializableDynamicNodeInvocation dynamic_node_invocation_to_serializable(
    DynamicNodeInvocation const &invocation) {
  return SerializableDynamicNodeInvocation{
      /*inputs=*/map_values(invocation.inputs,
                            dynamic_value_attrs_to_serializable),
      /*node_attrs=*/dynamic_node_attrs_to_serializable(invocation.node_attrs),
      /*outputs=*/
      map_values(invocation.outputs, dynamic_value_attrs_to_serializable),
  };
}

DynamicNodeInvocation dynamic_node_invocation_from_serializable(
    SerializableDynamicNodeInvocation const &invocation) {
  return DynamicNodeInvocation{
      /*inputs=*/map_values(invocation.inputs,
                            dynamic_value_attrs_from_serializable),
      /*node_attrs=*/
      dynamic_node_attrs_from_serializable(invocation.node_attrs),
      /*outputs=*/
      map_values(invocation.outputs, dynamic_value_attrs_from_serializable),
  };
}

} // namespace FlexFlow
