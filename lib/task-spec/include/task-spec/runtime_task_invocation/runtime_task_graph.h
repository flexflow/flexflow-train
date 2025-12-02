#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_TASK_INVOCATION_RUNTIME_TASK_GRAPH_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_TASK_INVOCATION_RUNTIME_TASK_GRAPH_H

#include "task-spec/runtime_task_invocation/runtime_task_graph.dtg.h"

namespace FlexFlow {

RuntimeTaskGraph
  fwd_runtime_task_graph_for_computation_graph(ComputationGraph const &);

RuntimeTaskGraph
  bwd_runtime_task_graph_for_computation_graph(ComputationGraph const &);

RuntimeTaskGraph
  update_runtime_task_graph_for_computation_graph(ComputationGraph const &);

} // namespace FlexFlow

#endif
