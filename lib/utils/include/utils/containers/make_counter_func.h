#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAKE_COUNTER_FUNC_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAKE_COUNTER_FUNC_H

#include <functional>

namespace FlexFlow {

std::function<int()> make_counter_func(int start = 0);

} // namespace FlexFlow

#endif
