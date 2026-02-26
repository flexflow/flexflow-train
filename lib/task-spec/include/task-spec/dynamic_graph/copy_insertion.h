#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_COPY_INSERTION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_COPY_INSERTION_H

#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

bool node_is_copy(DynamicNodeAttrs const &n);
bool value_is_mapped(DynamicValueAttrs const &);

bool no_part_of_graph_is_copy_inserted(DynamicOpenDataflowGraph const &);
bool graph_is_fully_copy_inserted(DynamicOpenDataflowGraph const &);

std::unordered_set<DynamicNodeInvocation> perform_copy_insertion_for_invocation(
    DynamicNodeInvocation const &i,
    std::unordered_map<DynamicValueAttrs, DynamicValueAttrs> const &sources);

DynamicOpenDataflowGraph
    perform_copy_insertion(DynamicOpenDataflowGraph const &);

} // namespace FlexFlow

#endif
