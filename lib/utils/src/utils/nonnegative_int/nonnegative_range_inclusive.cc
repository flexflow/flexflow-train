#include "utils/nonnegative_int/nonnegative_range_inclusive.h"
#include "utils/containers/range_inclusive.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

std::vector<nonnegative_int> nonnegative_range_inclusive(nonnegative_int start,
                                                         nonnegative_int end,
                                                         int step) {
  return transform(range_inclusive(start.unwrap_nonnegative(),
                                   end.unwrap_nonnegative(),
                                   step),
                   [](int x) { return nonnegative_int{x}; });
}

} // namespace FlexFlow
