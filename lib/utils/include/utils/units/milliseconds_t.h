#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNITS_MILLISECONDS_T_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNITS_MILLISECONDS_T_H

#include <functional>
#include <nlohmann/json.hpp>
#include <rapidcheck.h>
#include <string>

namespace FlexFlow {

struct milliseconds_t {
public:
  milliseconds_t() = delete;
  explicit milliseconds_t(float);

  bool operator<(milliseconds_t const &other) const;
  bool operator==(milliseconds_t const &other) const;
  bool operator>(milliseconds_t const &other) const;
  bool operator<=(milliseconds_t const &other) const;
  bool operator!=(milliseconds_t const &other) const;
  bool operator>=(milliseconds_t const &other) const;

  milliseconds_t operator+(milliseconds_t const &other) const;

  float unwrap_milliseconds() const;

private:
  float value;
};

milliseconds_t operator""_ms(long double);
milliseconds_t operator""_ms(unsigned long long int);

std::ostream &operator<<(std::ostream &, milliseconds_t const &);
std::string format_as(milliseconds_t const &);

} // namespace FlexFlow

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::milliseconds_t> {
  static ::FlexFlow::milliseconds_t from_json(json const &j);
  static void to_json(json &j, ::FlexFlow::milliseconds_t t);
};

} // namespace nlohmann

namespace rc {

template <>
struct Arbitrary<::FlexFlow::milliseconds_t> {
  static Gen<::FlexFlow::milliseconds_t> arbitrary();
};

} // namespace rc

namespace std {

template <>
struct hash<::FlexFlow::milliseconds_t> {
  size_t operator()(::FlexFlow::milliseconds_t const &) const noexcept;
};

} // namespace std

#endif
