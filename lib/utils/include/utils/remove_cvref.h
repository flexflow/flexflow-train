#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_REMOVE_CVREF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_REMOVE_CVREF_H

#include <type_traits>

namespace FlexFlow {

template <typename T>
using remove_cvref_t = typename std::remove_cv_t<std::remove_reference_t<T>>;

} // namespace FlexFlow

#endif
