#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ITERATOR_T_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ITERATOR_T_H

#include <iterator>

namespace FlexFlow {

template <typename T>
using iterator_t = decltype(std::begin(std::declval<T &>()));

} // namespace FlexFlow

#endif
