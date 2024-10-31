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
  explicit nonnegative_int(int const &value);
  explicit nonnegative_int(int &&value);

  explicit operator int() const noexcept;
  operator int &() noexcept;
  operator int const &() const noexcept;

  template <typename T,
            typename std::enable_if<(std::is_convertible<int, T>::value &&
                                     !std::is_same<int, T>::value),
                                    bool>::type = true>
  operator T() const {
    return (this->value_);
  }

  friend void swap(nonnegative_int &a, nonnegative_int &b) noexcept;

  bool operator<(nonnegative_int const &other);
  bool operator==(nonnegative_int const &other);
  bool operator>(nonnegative_int const &other);
  bool operator<=(nonnegative_int const &other);
  bool operator!=(nonnegative_int const &other);
  bool operator>=(nonnegative_int const &other);

  nonnegative_int &operator+=(nonnegative_int const &other);
  nonnegative_int &operator+=(int const &other);
  nonnegative_int &operator++();
  nonnegative_int operator++(int);
  nonnegative_int operator+(nonnegative_int const &other);
  nonnegative_int operator+(int const &other);
  friend nonnegative_int operator+(int const &lhs, nonnegative_int const &rhs);

  nonnegative_int &operator-=(nonnegative_int const &other);
  nonnegative_int &operator-=(int const &other);
  nonnegative_int &operator--();
  nonnegative_int operator--(int);
  nonnegative_int operator-(nonnegative_int const &other);
  nonnegative_int operator-(int const &other);
  friend nonnegative_int operator-(int const &lhs, nonnegative_int const &rhs);

  nonnegative_int &operator*=(nonnegative_int const &other);
  nonnegative_int &operator*=(int const &other);
  nonnegative_int operator*(nonnegative_int const &other);
  nonnegative_int operator*(int const &other);
  friend nonnegative_int operator*(int const &lhs, nonnegative_int const &rhs);

  nonnegative_int &operator/=(nonnegative_int const &other);
  nonnegative_int &operator/=(int const &other);
  nonnegative_int operator/(nonnegative_int const &other);
  nonnegative_int operator/(int const &other);
  friend nonnegative_int operator/(int const &lhs, nonnegative_int const &rhs);

  template <typename F>
  nonnegative_int fmap(F const &f) {
    static_assert(
        std::is_same<decltype(std::declval<F>()(std::declval<int const &>())),
                     int>::value,
        "Function must return an value of the underlying type");

    return nonnegative_int(f(this->value_));
  }

  friend std::ostream &operator<<(std::ostream &os, nonnegative_int const &n);

  int get_value() const;

private:
  int value_;
};
} // namespace FlexFlow

namespace nlohmann {
template <>
struct adl_serializer<::FlexFlow::nonnegative_int> {
  static ::FlexFlow::nonnegative_int from_json(json const &j) {
    return ::FlexFlow::nonnegative_int{j.template get<int>()};
  }
  static void to_json(json &j, ::FlexFlow::nonnegative_int t) {
    j = t.get_value();
  }
};
} // namespace nlohmann

namespace std {
template <>
struct hash<FlexFlow::nonnegative_int> {
  std::size_t operator()(FlexFlow::nonnegative_int const &n) const noexcept {
    return std::hash<int>{}(n.get_value());
  }
};
} // namespace std

#endif
