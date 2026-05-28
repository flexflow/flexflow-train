#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONEMPTY_SET_NONEMPTY_SET_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONEMPTY_SET_NONEMPTY_SET_H

#include <set>
#include <libassert/assert.hpp>
#include "utils/hash-utils.h"
#include "utils/hash/set.h"
#include "utils/fmt/set.h"
#include "utils/positive_int/positive_int.h"
#include "utils/containers/unordered_set_of.h"

namespace FlexFlow {

template <typename T>
struct nonempty_set {
public:
  nonempty_set() = delete;

  nonempty_set(std::initializer_list<T> const &vs) : raw(vs) {
    ASSERT(this->raw.size() > 0);
  }

  explicit nonempty_set(std::set<T> const &s) : raw(s) {
    ASSERT(this->raw.size() > 0);
  }

  bool operator==(nonempty_set const &other) const {
    return this->unwrap_as_set() == other.unwrap_as_set();
  }

  bool operator!=(nonempty_set const &other) const {
    return this->unwrap_as_set() != other.unwrap_as_set();
  }

  bool operator<(nonempty_set const &other) const {
    return this->unwrap_as_set() < other.unwrap_as_set();
  }

  bool operator<=(nonempty_set const &other) const {
    return this->unwrap_as_set() <= other.unwrap_as_set();
  }

  bool operator>(nonempty_set const &other) const {
    return this->unwrap_as_set() > other.unwrap_as_set();
  }

  bool operator>=(nonempty_set const &other) const {
    return this->unwrap_as_set() >= other.unwrap_as_set();
  }

  bool operator==(std::set<T> const &other) const {
    return this->unwrap_as_set() == other;
  }

  bool operator!=(std::set<T> const &other) const {
    return this->unwrap_as_set() != other;
  }

  void insert(T const &t) {
    this->raw.insert(t);
  }

  size_t size() const {
    return this->raw.size();
  };

  positive_int num_elements() const {
    return positive_int{this->raw.size()};
  };

  std::set<T> const &unwrap_as_set() const {
    return this->raw;
  }

  std::unordered_set<T> unwrap_as_unordered_set() const {
    return unordered_set_of(this->raw);
  }

  using value_type = T;

  typename std::set<T>::const_iterator begin() const {
    return this->raw.cbegin();
  }

  typename std::set<T>::const_iterator cbegin() const {
    return this->raw.cbegin();
  }

  typename std::set<T>::const_iterator end() const {
    return this->raw.cend();
  }

  typename std::set<T>::const_iterator cend() const {
    return this->raw.cend();
  }

private:
  std::set<T> raw;
};

template <typename T>
bool operator==(std::set<T> const &lhs,
                nonempty_set<T> const &rhs) {
  return lhs == rhs.unwrap_as_set();
}

template <typename T>
bool operator!=(std::set<T> const &lhs,
                nonempty_set<T> const &rhs) {
  return lhs != rhs.unwrap_as_set();
}

template <typename T>
std::set<T> format_as(nonempty_set<T> const &s) {
  return s.unwrap_as_set();
}

template <typename T>
std::ostream &operator<<(std::ostream &s, nonempty_set<T> const &m) {
  return (s << fmt::to_string(m));
}

} // namespace FlexFlow

namespace std {

template <typename T>
struct hash<::FlexFlow::nonempty_set<T>> {
  size_t operator()(::FlexFlow::nonempty_set<T> const &x) const {
    return ::FlexFlow::get_std_hash(x.unwrap_as_set());
  };
};

} // namespace std

#endif
