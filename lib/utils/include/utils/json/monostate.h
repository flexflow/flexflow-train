#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_MONOSTATE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_MONOSTATE_H

#include <nlohmann/json.hpp>
#include <variant>

namespace nlohmann {

template <>
struct adl_serializer<std::monostate> {
  static void to_json(json &, std::monostate);
  static void from_json(json const &, std::monostate &);
};

} // namespace FlexFlow

#endif
