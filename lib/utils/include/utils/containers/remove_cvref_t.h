#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REMOVE_CVREF_T_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REMOVE_CVREF_T_H

#include <iterator>

namespace FlexFlow {

template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

} // namespace FlexFlow

#endif
