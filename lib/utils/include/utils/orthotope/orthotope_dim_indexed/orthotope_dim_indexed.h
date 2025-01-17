#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_DIM_INDEXED_ORTHOTOPE_DIM_INDEXED_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_DIM_INDEXED_ORTHOTOPE_DIM_INDEXED_H

#include "utils/orthotope/orthotope_dim_idx_t.dtg.h"
#include <vector>
#include "utils/hash-utils.h"
#include <fmt/format.h>
#include "utils/hash/vector.h"
#include "utils/hash/tuple.h"
#include "utils/fmt/vector.h"
#include "utils/type_traits_core.h"
#include "utils/orthotope/orthotope_dim_idx_t.h"
#include <set>
#include "utils/ord/vector.h"

namespace FlexFlow {

template <typename T>
struct OrthotopeDimIndexed {
public:
  OrthotopeDimIndexed()
    : contents() 
  { }
  
  OrthotopeDimIndexed(std::initializer_list<T> const &l)
    : contents(l)
  { }

  template <typename Iter>
  OrthotopeDimIndexed(Iter begin, Iter end)
    : contents(begin, end)
  { }

  T const &at(orthotope_dim_idx_t const &idx) const {
    return this->contents.at(idx.raw_idx);
  }

  T &at(orthotope_dim_idx_t const &idx) {
    return this->contents.at(idx.raw_idx);
  }

  T const &back() const {
    return this->contents.back();
  }

  T &back() {
    return this->contents.back();
  }

  T const &front() const {
    return this->contents.front();
  }

  T &front() {
    return this->contents.front();
  }

  void push_back(T const &t) {
    this->contents.push_back(t);
  }

  std::vector<T> const &get_contents() const {
    return this->contents;
  }

  bool operator==(OrthotopeDimIndexed const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(OrthotopeDimIndexed const &other) const {
    return this->tie() != other.tie();
  }

  std::set<orthotope_dim_idx_t> indices() const {
    return dim_idxs_for_orthotope_with_num_dims(this->size());
  }

  std::tuple<std::vector<T> const &> tie() const {
    return std::tie(contents);
  }
private:
  std::vector<T> contents;
public:
  using iterator = typename decltype(contents)::iterator;
  using const_iterator = typename decltype(contents)::const_iterator;
  using reverse_iterator = typename decltype(contents)::reverse_iterator;
  using const_reverse_iterator = typename decltype(contents)::const_reverse_iterator;

  using value_type = typename decltype(contents)::value_type;
  using pointer = typename decltype(contents)::pointer;
  using const_pointer = typename decltype(contents)::const_pointer;
  using reference = typename decltype(contents)::reference;
  using const_reference = typename decltype(contents)::const_reference;

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
};

template <typename T>
std::enable_if_t<is_lt_comparable_v<T>, bool> operator<(OrthotopeDimIndexed<T> const &lhs, OrthotopeDimIndexed<T> const &rhs) {
  return lhs.tie() < rhs.tie();
}

template <typename T>
std::vector<T> format_as(OrthotopeDimIndexed<T> const &d) {
  return d.get_contents();
}

template <typename T>
std::ostream &operator<<(std::ostream &s, OrthotopeDimIndexed<T> const &d) {
  return (s << fmt::to_string(d));
}

} // namespace FlexFlow

namespace std {

template <typename T>
struct hash<::FlexFlow::OrthotopeDimIndexed<T>> {
  size_t operator()(::FlexFlow::OrthotopeDimIndexed<T> const &t) const {
    static_assert(::FlexFlow::is_hashable<T>::value,
                  "Elements must be hashable");

    return ::FlexFlow::get_std_hash(t.tie());
  }
};

} // namespace std

#endif
