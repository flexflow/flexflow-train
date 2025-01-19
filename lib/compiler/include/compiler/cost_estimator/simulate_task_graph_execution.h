#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_SIMULATE_TASK_GRAPH_EXECUTION_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_SIMULATE_TASK_GRAPH_EXECUTION_H

#include "compiler/cost_estimator/task_constraint.dtg.h"
#include "compiler/cost_estimator/task_graph.dtg.h"
#include "compiler/cost_estimator/task_graph_execution_trace.dtg.h"

namespace FlexFlow {

TaskGraphExecutionTrace
    simulate_task_graph_execution(TaskGraph const &task_graph,
                                  TaskConstraint const &constraint);

} // namespace FlexFlow

#endif
