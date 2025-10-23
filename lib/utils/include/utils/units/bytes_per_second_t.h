#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNITS_BYTES_PER_SECOND_T_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNITS_BYTES_PER_SECOND_T_H

#include "utils/units/milliseconds_t.h"
#include "utils/units/num_bytes_t.h"
#include <functional>
#include <nlohmann/json.hpp>
#include <rapidcheck.h>
#include <string>

namespace FlexFlow {

struct bytes_per_second_t {
public:
  bytes_per_second_t() = delete;
  explicit bytes_per_second_t(float);

  bool operator<(bytes_per_second_t const &other) const;
  bool operator==(bytes_per_second_t const &other) const;
  bool operator>(bytes_per_second_t const &other) const;
  bool operator<=(bytes_per_second_t const &other) const;
  bool operator!=(bytes_per_second_t const &other) const;
  bool operator>=(bytes_per_second_t const &other) const;

  bytes_per_second_t operator+(bytes_per_second_t const &other) const;

  friend milliseconds_t operator/(num_bytes_t, bytes_per_second_t);

  float unwrap_bytes_per_second() const;

private:
  float value;
};

std::ostream &operator<<(std::ostream &, bytes_per_second_t const &);
std::string format_as(bytes_per_second_t const &);

} // namespace FlexFlow

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::bytes_per_second_t> {
  static ::FlexFlow::bytes_per_second_t from_json(json const &j);
  static void to_json(json &j, ::FlexFlow::bytes_per_second_t t);
};

} // namespace nlohmann

namespace rc {

template <>
struct Arbitrary<::FlexFlow::bytes_per_second_t> {
  static Gen<::FlexFlow::bytes_per_second_t> arbitrary();
};

} // namespace rc

namespace std {

template <>
struct hash<::FlexFlow::bytes_per_second_t> {
  size_t operator()(::FlexFlow::bytes_per_second_t const &) const noexcept;
};

} // namespace std

#endif
