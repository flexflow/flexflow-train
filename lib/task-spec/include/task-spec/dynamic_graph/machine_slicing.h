#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_MACHINE_SLICING_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_MACHINE_SLICING_H

#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

std::unordered_set<DynamicNodeInvocation>
    perform_machine_slicing_for_invocation(DynamicNodeInvocation const &,
                                           MachineSpaceCoordinate const &);

DynamicOpenDataflowGraph
    perform_machine_slicing(DynamicOpenDataflowGraph const &,
                            MachineSpaceCoordinate const &);

} // namespace FlexFlow

#endif
