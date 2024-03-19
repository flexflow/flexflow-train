#ifndef _FLEXFLOW_LIB_UTILS_TYPE_INDEX_EXTRA_INCLUDE_UTILS_TYPE_INDEX_EXTRA_MATCHES_H
#define _FLEXFLOW_LIB_UTILS_TYPE_INDEX_EXTRA_INCLUDE_UTILS_TYPE_INDEX_EXTRA_MATCHES_H

#include "type_index.h"
#include <typeindex>

namespace FlexFlow {

template <typename T>
bool matches(std::type_index idx) {
  return idx == type_index<T>();
}

} // namespace FlexFlow

#endif