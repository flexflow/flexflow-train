#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_UPDATE_INSERTION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_UPDATE_INSERTION_H

#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

bool node_has_already_had_update_insertion_performed(DynamicNodeAttrs const &);
bool value_has_already_had_update_insertion_performed(DynamicValueAttrs const &);

bool node_is_ready_for_update_insertion(DynamicNodeAttrs const &);
bool value_is_ready_for_update_insertion(DynamicValueAttrs const &);

bool no_part_of_graph_has_had_update_insertion_performed(DynamicOpenDataflowGraph const &);
bool graph_is_ready_for_update_insertion(DynamicOpenDataflowGraph const &);

std::unordered_set<DynamicNodeInvocation>
    perform_update_insertion_for_invocation(DynamicNodeInvocation const &,
                                            OptimizerAttrs const &);

DynamicOpenDataflowGraph
    perform_update_insertion(DynamicOpenDataflowGraph const &,
                             OptimizerAttrs const &);

} // namespace FlexFlow

#endif
