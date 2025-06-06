#ifndef _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_FF_STACK_VECTOR_H
#define _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_FF_STACK_VECTOR_H

#include "op-attrs/ff_dim_t.dtg.h"
#include "op-attrs/relative_ff_dim_t.dtg.h"
#include "utils/containers/range.h"
#include "utils/fmt/vector.h"
#include "utils/stack_vector/stack_vector.h"
#include <nlohmann/json.hpp>

namespace FlexFlow {

template <typename Idx, typename T>
struct DimOrdered {
  DimOrdered() {}

  DimOrdered(std::initializer_list<T> const &l)
      : contents(l.begin(), l.end()) {}

  DimOrdered(std::vector<T> const &contents)
      : contents(contents.begin(), contents.end()) {}

  template <typename It>
  DimOrdered(It begin, It end) : contents(begin, end) {}

  template <size_t MAXSIZE>
  DimOrdered(stack_vector<T, MAXSIZE> const &contents)
      : contents(contents.begin(), contents.end()) {}

  T const &at(Idx idx) const {
    nonnegative_int raw = idx.value;
    return this->contents.at(raw.unwrap_nonnegative());
  }

  T &at(Idx idx) {
    nonnegative_int raw = idx.value;
    return this->contents.at(raw.unwrap_nonnegative());
  }

  T const &operator[](Idx idx) const {
    return this->at(idx);
  }

  T &operator[](Idx idx) {
    return this->at(idx);
  }

  bool idx_is_valid(Idx const &idx) const {
    nonnegative_int raw = idx.value;
    return (raw < this->contents.size());
  }

  bool operator==(DimOrdered const &other) const {
    return this->contents == other.contents;
  }

  bool operator!=(DimOrdered const &other) const {
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

  friend struct ::std::hash<DimOrdered>;

private:
  stack_vector<T, MAX_TENSOR_DIM> contents;
};

template <typename Idx, typename T>
auto operator<(DimOrdered<Idx, T> const &lhs, DimOrdered<Idx, T> const &rhs)
    -> std::enable_if_t<is_lt_comparable_v<T>, bool> {
  return std::lexicographical_compare(
      lhs.cbegin(), lhs.cend(), rhs.cbegin(), rhs.cend());
}

template <typename Idx, typename T>
std::string format_as(DimOrdered<Idx, T> const &v) {
  std::vector<T> as_vec(v.cbegin(), v.cend());
  return fmt::format("<ff_ordered {}>", as_vec);
}

template <typename Idx, typename T>
std::ostream &operator<<(std::ostream &s, DimOrdered<Idx, T> const &v) {
  return (s << fmt::to_string(v));
}

} // namespace FlexFlow

namespace nlohmann {
template <typename Idx, typename T>
struct adl_serializer<::FlexFlow::DimOrdered<Idx, T>> {
  static ::FlexFlow::DimOrdered<Idx, T> from_json(nlohmann::json const &j) {
    return {j.template get<std::vector<T>>()};
  }

  static void to_json(nlohmann::json &j,
                      ::FlexFlow::DimOrdered<Idx, T> const &x) {
    j = std::vector<T>{x.cbegin(), x.cend()};
  }
};
} // namespace nlohmann

namespace std {

template <typename Idx, typename T>
struct hash<::FlexFlow::DimOrdered<Idx, T>> {
  size_t operator()(::FlexFlow::DimOrdered<Idx, T> const &t) const {
    static_assert(::FlexFlow::is_hashable<T>::value,
                  "Elements must be hashable");

    return get_std_hash(t.contents);
  }
};

} // namespace std

namespace rc {

template <typename Idx, typename T>
struct Arbitrary<::FlexFlow::DimOrdered<Idx, T>> {
  static Gen<::FlexFlow::DimOrdered<Idx, T>> arbitrary() {
    return gen::construct<::FlexFlow::DimOrdered<Idx, T>>(
        gen::arbitrary<::FlexFlow::stack_vector<T, MAX_TENSOR_DIM>>());
  }
};

} // namespace rc

#endif
