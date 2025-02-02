#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/containers/range.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

std::vector<nonnegative_int> nonnegative_range(nonnegative_int end) {
  return transform(range(end.unwrap_nonnegative()),
                   [](int x) { return nonnegative_int{x}; });
}

std::vector<nonnegative_int>
    nonnegative_range(nonnegative_int start, nonnegative_int end, int step) {
  return transform(
      range(start.unwrap_nonnegative(), end.unwrap_nonnegative(), step),
      [](int x) { return nonnegative_int{x}; });
}

} // namespace FlexFlow
