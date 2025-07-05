#ifndef _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_DIM_ORDERED_FF_ORDERED_H
#define _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_DIM_ORDERED_FF_ORDERED_H

#include "op-attrs/ff_dim_t.dtg.h"
#include "op-attrs/relative_ff_dim_t.dtg.h"
#include "utils/fmt/vector.h"
#include "utils/stack_vector/stack_vector.h"

namespace FlexFlow {

template <typename T>
struct FFOrdered {
  FFOrdered() {}

  explicit FFOrdered(std::initializer_list<T> const &l) : contents(l.begin(), l.end()) {}

  template <typename It>
  explicit FFOrdered(It begin, It end) : contents(begin, end) {}

  template <size_t MAXSIZE>
  explicit FFOrdered(stack_vector<T, MAXSIZE> const &contents)
      : contents(contents.begin(), contents.end()) {}

  T const &at(ff_dim_t idx) const {
    int raw = idx.value.unwrap_nonnegative();
    return this->contents.at(raw);
  }

  T const &at(relative_ff_dim_t idx) const {
    int raw = idx.value;
    if (raw < 0) {
      raw = this->contents.size() + raw;
    }
    return this->contents.at(raw);
  }

  T &at(ff_dim_t idx) {
    int raw = idx.value.unwrap_nonnegative();
    return this->contents.at(raw);
  }

  T &at(relative_ff_dim_t idx) {
    int raw = idx.value;
    if (raw < 0) {
      raw = this->contents.size() + raw;
    }
    return this->contents.at(raw);
  }

  T const &operator[](ff_dim_t idx) const {
    return this->at(idx);
  }

  T const &operator[](relative_ff_dim_t idx) const {
    return this->at(idx);
  }

  T &operator[](ff_dim_t idx) {
    return this->at(idx);
  }

  T &operator[](relative_ff_dim_t idx) {
    return this->at(idx);
  }

  bool idx_is_valid(ff_dim_t const &idx) const {
    int raw = idx.value.unwrap_nonnegative();
    return raw < this->contents.size();
  }

  bool idx_is_valid(relative_ff_dim_t const &idx) const {
    int raw = idx.value;
    if (raw < 0) {
      raw = this->contents.size() + raw;
    }
    return (raw >= 0 && raw < this->contents.size());
  }

  bool operator==(FFOrdered const &other) const {
    return this->contents == other.contents;
  }

  bool operator!=(FFOrdered const &other) const {
    return this->contents != other.contents;
  }

  using iterator = typename stack_vector<T, MAX_TENSOR_DIM>::iterator;
  using const_iterator =
      typename stack_vector<T, MAX_TENSOR_DIM>::const_iterator;
  using reverse_iterator =
      typename stack_vector<T, MAX_TENSOR_DIM>::reverse_iterator;
  using const_reverse_iterator =
      typename stack_vector<T, MAX_TENSOR_DIM>::const_reverse_iterator;
  using value_type = T;
  using pointer = value_type *;
  using const_pointer = value_type const *;
  using reference = value_type &;
  using const_reference = value_type const &;

  iterator begin() {
    return this->contents.begin();
  }

  const_iterator begin() const {
    return this->cbegin();
  }

  const_iterator cbegin() const {
    return this->contents.cbegin();
  }

  iterator end() {
    return this->contents.end();
  }

  const_iterator end() const {
    return this->cend();
  }

  const_iterator cend() const {
    return this->contents.cend();
  }

  reverse_iterator rbegin() {
    return this->contents.rbegin();
  }

  const_reverse_iterator rbegin() const {
    return this->crbegin();
  }

  const_reverse_iterator crbegin() const {
    return this->contents.crbegin();
  }

  reverse_iterator rend() {
    return this->contents.rend();
  }

  const_reverse_iterator rend() const {
    return this->crend();
  }

  const_reverse_iterator crend() const {
    return this->contents.crend();
  }

  size_t size() const {
    return this->contents.size();
  }

  size_t empty() const {
    return this->contents.empty();
  }

  size_t num_dims() const {
    return this->size();
  }

  friend struct ::std::hash<FFOrdered>;

private:
  stack_vector<T, MAX_TENSOR_DIM> contents;
};

template <typename T>
auto operator<(FFOrdered<T> const &lhs, FFOrdered<T> const &rhs)
    -> std::enable_if_t<is_lt_comparable_v<T>, bool> {
  return std::lexicographical_compare(
      lhs.cbegin(), lhs.cend(), rhs.cbegin(), rhs.cend());
}

template <typename T>
std::string format_as(FFOrdered<T> const &v) {
  std::vector<T> as_vec(v.cbegin(), v.cend());
  return fmt::format("<ff_ordered {}>", as_vec);
}

template <typename T>
std::ostream &operator<<(std::ostream &s, FFOrdered<T> const &v) {
  return (s << fmt::to_string(v));
}

} // namespace FlexFlow

namespace nlohmann {
template <typename T>
struct adl_serializer<::FlexFlow::FFOrdered<T>> {
  static ::FlexFlow::FFOrdered<T> from_json(nlohmann::json const &j) {
    std::vector<T> v = j.template get<std::vector<T>>();
    return ::FlexFlow::FFOrdered<T>(v.cbegin(), v.cend());
  }

  static void to_json(nlohmann::json &j, ::FlexFlow::FFOrdered<T> const &x) {
    j = std::vector<T>{x.cbegin(), x.cend()};
  }
};
} // namespace nlohmann

namespace std {

template <typename T>
struct hash<::FlexFlow::FFOrdered<T>> {
  size_t operator()(::FlexFlow::FFOrdered<T> const &t) const {
    static_assert(::FlexFlow::is_hashable<T>::value,
                  "Elements must be hashable");

    return get_std_hash(t.contents);
  }
};

} // namespace std

namespace rc {

template <typename T>
struct Arbitrary<::FlexFlow::FFOrdered<T>> {
  static Gen<::FlexFlow::FFOrdered<T>> arbitrary() {
    return gen::construct<::FlexFlow::FFOrdered<T>>(
        gen::arbitrary<::FlexFlow::stack_vector<T, MAX_TENSOR_DIM>>());
  }
};

} // namespace rc

#endif
