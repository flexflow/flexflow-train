#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_CONCRETE_TASK_GRAPH_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_CONCRETE_TASK_GRAPH_H

#include "local-execution/local_concrete_task_graph.dtg.h"

namespace FlexFlow {

std::vector<LocalConcreteTaskInvocation>
  local_concrete_task_graph_topological_ordering(LocalConcreteTaskGraph const &);

} // namespace FlexFlow

#endif
