#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_RANGE_INCLUSIVE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_RANGE_INCLUSIVE_H

#include <vector>

namespace FlexFlow {

std::vector<int> range_inclusive(int start, int end, int step = 1);
std::vector<int> range_inclusive(int end);

} // namespace FlexFlow

#endif
