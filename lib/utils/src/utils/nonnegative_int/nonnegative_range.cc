#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/containers/range.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

std::vector<nonnegative_int> nonnegative_range(nonnegative_int end) {
  return transform(range(end.value()), 
                   [](int x) { return nonnegative_int{x}; });
}

std::vector<nonnegative_int> nonnegative_range(nonnegative_int start, nonnegative_int end, int step) {
  return transform(range(start.value(), end.value(), step), 
                   [](int x) { return nonnegative_int{x}; });
}

} // namespace FlexFlow
