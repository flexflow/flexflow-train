#include "utils/containers/range_inclusive.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

std::vector<int> range_inclusive(int start, int end, int step) {
  ASSERT(step != 0);

  std::vector<int> result;
  if (step > 0) {
    for (int i = start; i <= end; i += step) {
      result.push_back(i);
    }
  } else {
    for (int i = start; i >= end; i += step) {
      result.push_back(i);
    }
  }
  return result;
}

std::vector<int> range_inclusive(int end) {
  return range_inclusive(0, end);
}

} // namespace FlexFlow
