#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_SERIALIZABLE_DYNAMIC_NODE_ATTRS_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_SERIALIZABLE_DYNAMIC_NODE_ATTRS_H

#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_attrs.dtg.h"

namespace FlexFlow {

SerializableDynamicNodeAttrs
    dynamic_node_attrs_to_serializable(DynamicNodeAttrs const &);
DynamicNodeAttrs
    dynamic_node_attrs_from_serializable(SerializableDynamicNodeAttrs const &);

} // namespace FlexFlow

#endif
