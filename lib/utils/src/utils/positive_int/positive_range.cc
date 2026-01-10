#include "utils/positive_int/positive_range.h"
#include "utils/containers/transform.h"
#include "utils/containers/range.h"

namespace FlexFlow {

std::vector<positive_int>
    positive_range(positive_int start, positive_int end, int step) {
  return transform(
      range(start.int_from_positive_int(), end.int_from_positive_int(), step),
      [](int x) { return positive_int{x}; });
}

} // namespace FlexFlow
