#include "compiler/task_graph_simulator/in_progress_task_comparator.h"
#include <tuple>

namespace FlexFlow {

bool InProgressTaskComparator::operator()(InProgressTask const &lhs,
                                          InProgressTask const &rhs) const {
  return std::tie(lhs.end_time, lhs.node) > std::tie(rhs.end_time, rhs.node);
}

} // namespace FlexFlow
