#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_TASK_GRAPH_TRAVERSAL_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_TASK_GRAPH_TRAVERSAL_H

#include "compiler/cost_estimator/task_graph.dtg.h"

namespace FlexFlow {

float simulate_forward_pass(TaskGraph const &task_graph);

} // namespace FlexFlow

#endif
