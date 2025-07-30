#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNITS_NUM_BYTES_T_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNITS_NUM_BYTES_T_H

#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

struct num_bytes_t {
public:
  num_bytes_t() = delete;
  explicit num_bytes_t(nonnegative_int);

  bool operator<(num_bytes_t const &other) const;
  bool operator==(num_bytes_t const &other) const;
  bool operator>(num_bytes_t const &other) const;
  bool operator<=(num_bytes_t const &other) const;
  bool operator!=(num_bytes_t const &other) const;
  bool operator>=(num_bytes_t const &other) const;

  num_bytes_t operator+(num_bytes_t const &other) const;

  nonnegative_int unwrap_num_bytes() const;

private:
  nonnegative_int value;
};

num_bytes_t operator""_bytes(unsigned long long int);

std::ostream &operator<<(std::ostream &, num_bytes_t const &);
std::string format_as(num_bytes_t const &);

} // namespace FlexFlow

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::num_bytes_t> {
  static ::FlexFlow::num_bytes_t from_json(json const &j);
  static void to_json(json &j, ::FlexFlow::num_bytes_t t);
};

} // namespace nlohmann

namespace rc {

template <>
struct Arbitrary<::FlexFlow::num_bytes_t> {
  static Gen<::FlexFlow::num_bytes_t> arbitrary();
};

} // namespace rc

namespace std {

template <>
struct hash<::FlexFlow::num_bytes_t> {
  size_t operator()(::FlexFlow::num_bytes_t const &) const noexcept;
};

} // namespace std

#endif
