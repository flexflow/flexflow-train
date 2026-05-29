#include "task-spec/dynamic_graph/dynamic_node_invocation.h"
#include "utils/containers/values.h"
#include "utils/containers/keys.h"
#include "utils/containers/all_of.h"

namespace FlexFlow {

bool invocation_fully_satisfies(DynamicNodeInvocation const &i,
                                std::function<bool(DynamicNodeAttrs const &)> const &node_condition,
                                std::function<bool(DynamicValueAttrs const &)> const &value_condition,
                                std::function<bool(DynamicTensorSlot const &)> const &slot_condition)
{
  return node_condition(i.node_attrs)
    && all_of(values(i.inputs), value_condition)
    && all_of(keys(i.inputs), slot_condition)
    && all_of(values(i.outputs), value_condition)
    && all_of(keys(i.outputs), slot_condition);
}

void require_invocation_fully_satisfies(DynamicNodeInvocation const &i,
                                        std::function<void(DynamicNodeAttrs const &)> const &require_node_condition,
                                        std::function<void(DynamicValueAttrs const &)> const &require_value_condition,
                                        std::function<void(DynamicTensorSlot const &)> const &require_slot_condition) {
  require_node_condition(i.node_attrs);
  for (DynamicTensorSlot const &k : keys(i.inputs)) {
    require_slot_condition(k);
    require_value_condition(i.inputs.at(k));
  }
  for (DynamicTensorSlot const &k : keys(i.outputs)) {
    require_slot_condition(k);
    require_value_condition(i.outputs.at(k));
  }
}

} // namespace FlexFlow
