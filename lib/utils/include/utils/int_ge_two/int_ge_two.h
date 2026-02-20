#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_INT_GE_TWO_INT_GE_TWO_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_INT_GE_TWO_INT_GE_TWO_H

#include "utils/positive_int/positive_int.h"
#include <optional>

namespace FlexFlow {

struct int_ge_two {
  int_ge_two() = delete;
  explicit int_ge_two(int value);
  explicit int_ge_two(size_t value);
  explicit int_ge_two(nonnegative_int value);
  explicit int_ge_two(positive_int value);

  explicit operator int() const noexcept;
  explicit operator nonnegative_int() const noexcept;
  explicit operator positive_int() const noexcept;

  bool operator<(int_ge_two other) const;
  bool operator==(int_ge_two other) const;
  bool operator>(int_ge_two other) const;
  bool operator<=(int_ge_two other) const;
  bool operator!=(int_ge_two other) const;
  bool operator>=(int_ge_two other) const;

  bool operator<(positive_int other) const;
  bool operator==(positive_int other) const;
  bool operator>(positive_int other) const;
  bool operator<=(positive_int other) const;
  bool operator!=(positive_int other) const;
  bool operator>=(positive_int other) const;

  friend bool operator<(positive_int lhs, int_ge_two rhs);
  friend bool operator==(positive_int lhs, int_ge_two rhs);
  friend bool operator>(positive_int lhs, int_ge_two rhs);
  friend bool operator<=(positive_int lhs, int_ge_two rhs);
  friend bool operator!=(positive_int lhs, int_ge_two rhs);
  friend bool operator>=(positive_int lhs, int_ge_two rhs);

  bool operator<(nonnegative_int other) const;
  bool operator==(nonnegative_int other) const;
  bool operator>(nonnegative_int other) const;
  bool operator<=(nonnegative_int other) const;
  bool operator!=(nonnegative_int other) const;
  bool operator>=(nonnegative_int other) const;

  friend bool operator<(nonnegative_int lhs, int_ge_two rhs);
  friend bool operator==(nonnegative_int lhs, int_ge_two rhs);
  friend bool operator>(nonnegative_int lhs, int_ge_two rhs);
  friend bool operator<=(nonnegative_int lhs, int_ge_two rhs);
  friend bool operator!=(nonnegative_int lhs, int_ge_two rhs);
  friend bool operator>=(nonnegative_int lhs, int_ge_two rhs);

  bool operator<(int other) const;
  bool operator==(int other) const;
  bool operator>(int other) const;
  bool operator<=(int other) const;
  bool operator!=(int other) const;
  bool operator>=(int other) const;

  friend bool operator<(int lhs, int_ge_two rhs);
  friend bool operator==(int lhs, int_ge_two rhs);
  friend bool operator>(int lhs, int_ge_two rhs);
  friend bool operator<=(int lhs, int_ge_two rhs);
  friend bool operator!=(int lhs, int_ge_two rhs);
  friend bool operator>=(int lhs, int_ge_two rhs);

  int_ge_two operator+(int_ge_two other) const;
  int_ge_two operator+(positive_int other) const;
  int_ge_two operator+(nonnegative_int other) const;
  int_ge_two &operator++();
  int_ge_two operator++(int);
  int_ge_two &operator+=(int_ge_two other);
  int_ge_two &operator+=(positive_int other);
  int_ge_two &operator+=(nonnegative_int other);

  friend int_ge_two operator+(nonnegative_int lhs, int_ge_two rhs);
  friend int_ge_two operator+(positive_int lhs, int_ge_two rhs);

  int_ge_two operator*(int_ge_two other) const;
  int_ge_two &operator*=(int_ge_two other);
  int_ge_two operator*(positive_int other) const;
  int_ge_two &operator*=(positive_int other);
  nonnegative_int operator*(nonnegative_int other) const;

  friend int_ge_two operator*(positive_int lhs, int_ge_two rhs);
  friend nonnegative_int operator*(nonnegative_int lhs, int_ge_two rhs);

  int int_from_int_ge_two() const;
  nonnegative_int nonnegative_int_from_int_ge_two() const;
  positive_int positive_int_from_int_ge_two() const;

  friend std::ostream &operator<<(std::ostream &os, int_ge_two n);

  friend int format_as(int_ge_two);

private:
  void check_invariant() const;

private:
  int value_;
};

int_ge_two operator""_ge2(unsigned long long int);

std::optional<int_ge_two> try_int_ge_two_from_positive_int(positive_int);

} // namespace FlexFlow

namespace nlohmann {
template <>
struct adl_serializer<::FlexFlow::int_ge_two> {
  static ::FlexFlow::int_ge_two from_json(json const &j);
  static void to_json(json &j, ::FlexFlow::int_ge_two t);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<::FlexFlow::int_ge_two> {
  static Gen<::FlexFlow::int_ge_two> arbitrary();
};
} // namespace rc

namespace std {
template <>
struct hash<::FlexFlow::int_ge_two> {
  std::size_t operator()(FlexFlow::int_ge_two n) const noexcept;
};
} // namespace std
#endif
