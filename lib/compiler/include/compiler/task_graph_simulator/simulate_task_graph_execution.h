#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_SIMULATE_TASK_GRAPH_EXECUTION_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_SIMULATE_TASK_GRAPH_EXECUTION_H

#include "compiler/task_graph_simulator/task_execution_constraint.dtg.h"
#include "compiler/task_graph_simulator/task_graph.dtg.h"
#include "compiler/task_graph_simulator/task_graph_execution_trace.dtg.h"

namespace FlexFlow {

TaskGraphExecutionTrace
    simulate_task_graph_execution(TaskGraph const &task_graph,
                                  TaskExecutionConstraint const &constraint);

} // namespace FlexFlow

#endif
