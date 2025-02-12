#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONNEGATIVE_INT_NONNEGATIVE_INT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONNEGATIVE_INT_NONNEGATIVE_INT_H

#include <any>
#include <fmt/format.h>
#include <functional>
#include <nlohmann/json.hpp>
#include <rapidcheck.h>
#include <string>

namespace FlexFlow {
class nonnegative_int {
public:
  nonnegative_int() = delete;
  explicit nonnegative_int(int value);
  explicit nonnegative_int(size_t value);

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

  nonnegative_int operator+(nonnegative_int const &other) const;
  nonnegative_int &operator++();
  nonnegative_int operator++(int);
  nonnegative_int &operator+=(nonnegative_int const &other);

  nonnegative_int operator*(nonnegative_int const &other) const;
  nonnegative_int &operator*=(nonnegative_int const &other);

  nonnegative_int operator/(nonnegative_int const &other) const;
  nonnegative_int &operator/=(nonnegative_int const &other);

  friend float operator/(float lhs, nonnegative_int rhs);
  friend float &operator/=(float &lhs, nonnegative_int rhs);

  nonnegative_int operator%(nonnegative_int const &other) const;
  nonnegative_int &operator%=(nonnegative_int const &other);

  friend std::ostream &operator<<(std::ostream &os, nonnegative_int const &n);

  friend int format_as(nonnegative_int const &);

  int unwrap_nonnegative() const;

private:
  int value_;
};

nonnegative_int operator""_n(unsigned long long int);

} // namespace FlexFlow

namespace nlohmann {
template <>
struct adl_serializer<::FlexFlow::nonnegative_int> {
  static ::FlexFlow::nonnegative_int from_json(json const &j);
  static void to_json(json &j, ::FlexFlow::nonnegative_int t);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<::FlexFlow::nonnegative_int> {
  static Gen<::FlexFlow::nonnegative_int> arbitrary();
};
} // namespace rc

namespace std {
template <>
struct hash<::FlexFlow::nonnegative_int> {
  std::size_t operator()(FlexFlow::nonnegative_int const &n) const noexcept;
};
} // namespace std

#endif
