#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_IN_PROGRESS_TASK_COMPARATOR_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_IN_PROGRESS_TASK_COMPARATOR_H

#include "compiler/task_graph_simulator/in_progress_task.dtg.h"
#include <tuple>

namespace FlexFlow {
struct InProgressTaskComparator {
  bool operator()(InProgressTask const &lhs, InProgressTask const &rhs) const;
};
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_IN_PROGRESS_TASK_COMPARATOR_H
