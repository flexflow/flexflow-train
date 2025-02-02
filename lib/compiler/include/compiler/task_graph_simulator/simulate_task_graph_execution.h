#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_SIMULATE_TASK_GRAPH_EXECUTION_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_SIMULATE_TASK_GRAPH_EXECUTION_H

#include "compiler/task_graph_simulator/task_execution_constraint.dtg.h"
#include "compiler/task_graph_simulator/task_graph_execution_trace.dtg.h"
#include "utils/graph/digraph/digraph_view.h"
#include <functional>
namespace FlexFlow {

TaskGraphExecutionTrace simulate_task_graph_execution(
    DiGraphView const &task_graph,
    std::function<float(Node const &)> cost_function,
    TaskExecutionConstraint const &constraint);

} // namespace FlexFlow

#endif
