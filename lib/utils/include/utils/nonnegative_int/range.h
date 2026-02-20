#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONNEGATIVE_INT_RANGE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONNEGATIVE_INT_RANGE_H

#include "utils/nonnegative_int/nonnegative_int.h"
#include <vector>

namespace FlexFlow {

std::vector<nonnegative_int>
    range(nonnegative_int start, nonnegative_int end, int step = 1);
std::vector<nonnegative_int> range(nonnegative_int end);

} // namespace FlexFlow

#endif
