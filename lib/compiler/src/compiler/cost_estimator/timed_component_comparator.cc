#include "compiler/cost_estimator/timed_component_comparator.h"
#include "utils/overload.h"

namespace FlexFlow {

bool TimedComponentComparator::operator()(TimedComponent const &lhs,
                                          TimedComponent const &rhs) const {
  float lhs_endtime = lhs.visit<float>(
      overload{[](TimedLayer const &layer) { return layer.endtime; },
               [](TimedDependency const &dep) { return dep.endtime; }});

  float rhs_endtime = rhs.visit<float>(
      overload{[](TimedLayer const &layer) { return layer.endtime; },
               [](TimedDependency const &dep) { return dep.endtime; }});

  return lhs_endtime > rhs_endtime;
}

} // namespace FlexFlow
