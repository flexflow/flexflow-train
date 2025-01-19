#include "compiler/cost_estimator/in_progress_task_comparator.h"
#include <tuple>

namespace FlexFlow {

bool InProgressTaskComparator::operator()(InProgressTask const &lhs,
                                          InProgressTask const &rhs) const {
  return std::tie(lhs.endtime, lhs.node) > std::tie(rhs.endtime, rhs.node);
}

} // namespace FlexFlow
