#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONNEGATIVE_INT_NONNEGATIVE_INT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONNEGATIVE_INT_NONNEGATIVE_INT_H

#include "rapidcheck.h"

#include <any>
#include <fmt/format.h>
#include <functional>
#include <nlohmann/json.hpp>
#include <string>

namespace FlexFlow {
class nonnegative_int {
public:
  nonnegative_int() = delete;
  explicit nonnegative_int(int value);

  explicit operator int() const noexcept;

  bool operator<(nonnegative_int const &other) const;
  bool operator==(nonnegative_int const &other) const;
  bool operator>(nonnegative_int const &other) const;
  bool operator<=(nonnegative_int const &other) const;
  bool operator!=(nonnegative_int const &other) const;
  bool operator>=(nonnegative_int const &other) const;

  bool operator<(int const &other) const;
  bool operator==(int const &other) const;
  bool operator>(int const &other) const;
  bool operator<=(int const &other) const;
  bool operator!=(int const &other) const;
  bool operator>=(int const &other) const;

  friend bool operator<(int const &lhs, nonnegative_int const &rhs);
  friend bool operator==(int const &lhs, nonnegative_int const &rhs);
  friend bool operator>(int const &lhs, nonnegative_int const &rhs);
  friend bool operator<=(int const &lhs, nonnegative_int const &rhs);
  friend bool operator!=(int const &lhs, nonnegative_int const &rhs);
  friend bool operator>=(int const &lhs, nonnegative_int const &rhs);

  friend std::ostream &operator<<(std::ostream &os, nonnegative_int const &n);

  friend int format_as(nonnegative_int const &);

  int get_value() const;

private:
  int value_;
};
} // namespace FlexFlow

namespace nlohmann {
template <>
struct adl_serializer<::FlexFlow::nonnegative_int> {
  static ::FlexFlow::nonnegative_int from_json(json const &j);
  static void to_json(json &j, ::FlexFlow::nonnegative_int t);
};
} // namespace nlohmann

namespace std {
template <>
struct hash<::FlexFlow::nonnegative_int> {
  std::size_t operator()(FlexFlow::nonnegative_int const &n) const noexcept;
};
} // namespace std

#endif