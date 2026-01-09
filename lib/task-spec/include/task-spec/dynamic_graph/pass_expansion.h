#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_PASS_EXPANSION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_PASS_EXPANSION_H

#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

bool node_is_pass_expanded(DynamicNodeAttrs const &);
bool value_is_pass_expanded(DynamicValueAttrs const &);
bool slot_is_pass_expanded(DynamicTensorSlot const &);

bool no_part_of_graph_is_pass_expanded(DynamicOpenDataflowGraph const &);
bool graph_is_fully_pass_expanded(DynamicOpenDataflowGraph const &);

DynamicNodeInvocation
    perform_fwd_pass_expansion_for_invocation(DynamicNodeInvocation const &);
DynamicNodeInvocation
    perform_bwd_pass_expansion_for_invocation(DynamicNodeInvocation const &);

// pass expansion here
DynamicOpenDataflowGraph
    perform_pass_expansion(DynamicOpenDataflowGraph const &);

} // namespace FlexFlow

#endif
