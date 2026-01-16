#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_POSITIVE_INT_POSITIVE_RANGE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_POSITIVE_INT_POSITIVE_RANGE_H

#include "utils/positive_int/positive_int.h"

namespace FlexFlow {

std::vector<positive_int>
    positive_range(positive_int start, positive_int end, int step = 1);

} // namespace FlexFlow

#endif
