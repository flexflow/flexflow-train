#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_NODE_INVOCATION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_NODE_INVOCATION_H

#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"

namespace FlexFlow {

bool invocation_fully_satisfies_expansion_conditions(
  std::function<bool(DynamicNodeAttrs const &)> const &node_condition,
  std::function<bool(DynamicTensorSlot const &)> const &slot_condition,
  std::function<bool(DynamicTensorSlotArguments const &)> const &) {

]


} // namespace FlexFlow

#endif
