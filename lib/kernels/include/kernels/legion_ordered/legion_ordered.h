#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LEGION_ORDERED_LEGION_ORDERED_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LEGION_ORDERED_LEGION_ORDERED_H

#include "kernels/legion_dim_t.dtg.h"
#include "utils/fmt/vector.h"
#include "utils/stack_vector/stack_vector.h"

namespace FlexFlow {

template <typename T>
struct LegionOrdered {
  LegionOrdered() {}

  explicit LegionOrdered(std::initializer_list<T> const &l)
      : contents(l.begin(), l.end()) {}

  explicit LegionOrdered(std::vector<T> const &contents)
      : contents(contents.begin(), contents.end()) {}

  template <typename It>
  explicit LegionOrdered(It begin, It end) : contents(begin, end) {}

  template <size_t MAXSIZE>
  explicit LegionOrdered(stack_vector<T, MAXSIZE> const &contents)
      : contents(contents.begin(), contents.end()) {}

  T const &at(legion_dim_t idx) const {
    int raw = idx.value.unwrap_nonnegative();
    return this->contents.at(raw);
  }

  T &at(legion_dim_t idx) {
    int raw = idx.value.unwrap_nonnegative();
    return this->contents.at(raw);
  }

  T const &operator[](legion_dim_t idx) const {
    return this->at(idx);
  }

  T &operator[](legion_dim_t idx) {
    return this->at(idx);
  }

  bool idx_is_valid(legion_dim_t const &idx) const {
    int raw = idx.value.unwrap_nonnegative();
    return raw < this->contents.size();
  }

  bool operator==(LegionOrdered const &other) const {
    return this->contents == other.contents;
  }

  bool operator!=(LegionOrdered const &other) const {
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

  friend struct ::std::hash<LegionOrdered>;

private:
  stack_vector<T, MAX_TENSOR_DIM> contents;
};

template <typename T>
auto operator<(LegionOrdered<T> const &lhs, LegionOrdered<T> const &rhs)
    -> std::enable_if_t<is_lt_comparable_v<T>, bool> {
  return std::lexicographical_compare(
      lhs.cbegin(), lhs.cend(), rhs.cbegin(), rhs.cend());
}

template <typename T>
std::string format_as(LegionOrdered<T> const &v) {
  std::vector<T> as_vec(v.cbegin(), v.cend());
  return fmt::format("<legion_ordered {}>", as_vec);
}

template <typename T>
std::ostream &operator<<(std::ostream &s, LegionOrdered<T> const &v) {
  return (s << fmt::to_string(v));
}

} // namespace FlexFlow

namespace nlohmann {
template <typename T>
struct adl_serializer<::FlexFlow::LegionOrdered<T>> {
  static ::FlexFlow::LegionOrdered<T> from_json(nlohmann::json const &j) {
    return {j.template get<std::vector<T>>()};
  }

  static void to_json(nlohmann::json &j,
                      ::FlexFlow::LegionOrdered<T> const &x) {
    j = std::vector<T>{x.cbegin(), x.cend()};
  }
};
} // namespace nlohmann

namespace std {

template <typename T>
struct hash<::FlexFlow::LegionOrdered<T>> {
  size_t operator()(::FlexFlow::LegionOrdered<T> const &t) const {
    static_assert(::FlexFlow::is_hashable<T>::value,
                  "Elements must be hashable");

    return get_std_hash(t.contents);
  }
};

} // namespace std

namespace rc {

template <typename T>
struct Arbitrary<::FlexFlow::LegionOrdered<T>> {
  static Gen<::FlexFlow::LegionOrdered<T>> arbitrary() {
    return gen::construct<::FlexFlow::LegionOrdered<T>>(
        gen::arbitrary<::FlexFlow::stack_vector<T, MAX_TENSOR_DIM>>());
  }
};

} // namespace rc

#endif
