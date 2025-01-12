#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_TIMED_COMPONENT_COMPARATOR_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_TIMED_COMPONENT_COMPARATOR_H

#include "compiler/cost_estimator/timed_component.dtg.h"

namespace FlexFlow {

struct TimedComponentComparator {
  bool operator()(TimedComponent const &lhs, TimedComponent const &rhs) const;
};

} // namespace FlexFlow

#endif
