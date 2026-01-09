#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_UPDATE_INSERTION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_UPDATE_INSERTION_H

#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

std::unordered_set<DynamicNodeInvocation>
    perform_update_insertion_for_invocation(DynamicNodeInvocation const &,
                                            OptimizerAttrs const &);

// after backwards pass, update tasks apply gradients to the weights (and this
// inserts the corresponding task)
DynamicOpenDataflowGraph
    perform_update_insertion(DynamicOpenDataflowGraph const &,
                             OptimizerAttrs const &);

} // namespace FlexFlow

#endif
