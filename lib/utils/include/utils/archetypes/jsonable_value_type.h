#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ARCHETYPES_JSONABLE_VALUE_TYPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ARCHETYPES_JSONABLE_VALUE_TYPE_H

#include <fmt/format.h>
#include <functional>
#include <libassert/assert.hpp>
#include <nlohmann/json.hpp>
#include <ostream>
#include <sstream>

namespace FlexFlow {

template <int TAG>
struct jsonable_value_type {
  jsonable_value_type() = delete;

  jsonable_value_type(jsonable_value_type const &) {
    PANIC();
  }
  jsonable_value_type &operator=(jsonable_value_type const &) {
    PANIC();
  }

  jsonable_value_type(jsonable_value_type &&) {
    PANIC();
  }
  jsonable_value_type &operator=(jsonable_value_type &&) {
    PANIC();
  }

  bool operator==(jsonable_value_type const &) const {
    PANIC();
  }
  bool operator!=(jsonable_value_type const &) const {
    PANIC();
  }
};

template <int TAG>
std::string format_as(jsonable_value_type<TAG> const &) {
  PANIC();
}

template <int TAG>
std::ostream &operator<<(std::ostream &s, jsonable_value_type<TAG> const &x) {
  PANIC();
}

} // namespace FlexFlow

namespace nlohmann {

template <int TAG>
struct adl_serializer<::FlexFlow::jsonable_value_type<TAG>> {
  static ::FlexFlow::jsonable_value_type<TAG> from_json(json const &) {
    PANIC();
  }

  static void to_json(json &, ::FlexFlow::jsonable_value_type<TAG> const &) {
    PANIC();
  }
};

} // namespace nlohmann

namespace std {

template <int TAG>
struct hash<::FlexFlow::jsonable_value_type<TAG>> {
  size_t operator()(::FlexFlow::jsonable_value_type<TAG> const &) const {
    PANIC();
  };
};

} // namespace std

#endif
