#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONNEGATIVE_INT_NONNEGATIVE_RANGE_INCLUSIVE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONNEGATIVE_INT_NONNEGATIVE_RANGE_INCLUSIVE_H

#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

std::vector<nonnegative_int> nonnegative_range_inclusive(nonnegative_int start,
                                                         nonnegative_int end,
                                                         int step = 1);

} // namespace FlexFlow

#endif
