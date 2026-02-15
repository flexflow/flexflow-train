#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_SERIALIZABLE_DYNAMIC_NODE_INVOCATION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_SERIALIZABLE_DYNAMIC_NODE_INVOCATION_H

#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.dtg.h"

namespace FlexFlow {

SerializableDynamicNodeInvocation
    dynamic_node_invocation_to_serializable(DynamicNodeInvocation const &);
DynamicNodeInvocation dynamic_node_invocation_from_serializable(
    SerializableDynamicNodeInvocation const &);

} // namespace FlexFlow

#endif
