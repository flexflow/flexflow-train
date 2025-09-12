#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_CHECK_IS_JSON_SERIALIZABLE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_CHECK_IS_JSON_SERIALIZABLE_H

#include "utils/json/is_json_serializable.h"

namespace FlexFlow {

#define CHECK_IS_JSON_SERIALIZABLE(...)                                   \
  static_assert(::FlexFlow::is_json_serializable<__VA_ARGS__>::value,             \
                #__VA_ARGS__ " should be json serializeable")

} // namespace FlexFlow

#endif
