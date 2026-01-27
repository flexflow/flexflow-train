#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_HALF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_HALF_H

#include "utils/half.h"
#include <nlohmann/json.hpp>

namespace nlohmann {

template <>
struct adl_serializer<half> {
  static void to_json(json &j, half x);
  static void from_json(json const &j, half &t);
};

} // namespace nlohmann

#endif
