#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_NODE_INVOCATION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_NODE_INVOCATION_H

#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"

namespace FlexFlow {

bool invocation_fully_satisfies(DynamicNodeInvocation const &,
                                std::function<bool(DynamicNodeAttrs const &)> const &node_condition,
                                std::function<bool(DynamicValueAttrs const &)> const &value_condition,
                                std::function<bool(DynamicTensorSlot const &)> const &slot_condition);

void require_invocation_fully_satisfies(DynamicNodeInvocation const &,
                                        std::function<void(DynamicNodeAttrs const &)> const &require_node_condition,
                                        std::function<void(DynamicValueAttrs const &)> const &require_value_condition,
                                        std::function<void(DynamicTensorSlot const &)> const &require_slot_condition);

} // namespace FlexFlow

#endif
