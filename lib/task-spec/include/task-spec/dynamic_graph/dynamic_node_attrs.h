#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_NODE_ATTRS_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_NODE_ATTRS_H

#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_mapping.dtg.h"

namespace FlexFlow {

DynamicNodeAttrs decide_dynamic_node_attrs_mapping(DynamicNodeAttrs const &,
                                                   DynamicNodeMapping const &);

} // namespace FlexFlow

#endif
