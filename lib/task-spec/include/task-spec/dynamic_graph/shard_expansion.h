#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_SHARD_EXPANSION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_SHARD_EXPANSION_H

#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

bool node_is_shard_expanded(DynamicNodeAttrs const &);
bool value_is_shard_expanded(DynamicValueAttrs const &);

bool no_part_of_graph_is_shard_expanded(DynamicOpenDataflowGraph const &);
bool graph_is_fully_shard_expanded(DynamicOpenDataflowGraph const &);

std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_invocation(DynamicNodeInvocation const &);

DynamicOpenDataflowGraph
    perform_shard_expansion(DynamicOpenDataflowGraph const &);

} // namespace FlexFlow

#endif
