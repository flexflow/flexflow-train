#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_MANY_TO_ONE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_MANY_TO_ONE_H

#include <unordered_set>
#include <unordered_map>
#include "utils/containers/try_at.h"
#include <fmt/format.h>
#include "utils/hash-utils.h"
#include "utils/hash/unordered_map.h"
#include "utils/hash/unordered_set.h"
#include "utils/hash/tuple.h"
#include "utils/containers/keys.h"
#include "utils/exception.h"
#include "utils/fmt/unordered_map.h"
#include "utils/fmt/unordered_set.h"

namespace FlexFlow {

template <typename L, typename R>
struct ManyToOne {
public:
  ManyToOne()
    : l_to_r(), r_to_l()
  { }

  template <typename It>
  ManyToOne(It start, It end) 
    : ManyToOne()
  { 
    for (; start < end; start++) {
      for (L const &l : start->first) {
        this->insert(std::pair<L, R>{l, start->second});
      }
    }
  }

  ManyToOne(std::initializer_list<std::pair<std::initializer_list<L>, R>> const &l_to_r)
    : ManyToOne(l_to_r.begin(), l_to_r.end())
  { }

  bool operator==(ManyToOne const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(ManyToOne const &other) const {
    return this->tie() != other.tie();
  }

  void insert(std::pair<L, R> const &p) {
    L l = p.first;
    R r = p.second;

    std::optional<R> found_r = try_at(this->l_to_r, l);

    if (!found_r.has_value()) {
      this->l_to_r.insert({l, r});
      this->r_to_l[r].insert(l);
    } else if (found_r.value() == r) {
      return;
    } else {
      throw mk_runtime_error(fmt::format("Existing mapping found for left value {}: tried to map to right value {}, but is already bound to right value {}", l, r, found_r.value()));
    }
  }

  R const &at_l(L const &l) const {
    return this->l_to_r.at(l);
  }

  std::unordered_set<L> const &at_r(R const &r) const {
    return this->r_to_l.at(r);
  }

  std::unordered_set<L> left_values() const {
    return keys(this->l_to_r);
  }

  std::unordered_set<R> right_values() const {
    return keys(this->r_to_l);
  }
private: 
  std::unordered_map<L, R> l_to_r;
  std::unordered_map<R, std::unordered_set<L>> r_to_l;
private:
  std::tuple<decltype(l_to_r) const &, decltype(r_to_l) const &> tie() const {
    return std::tie(this->l_to_r, this->r_to_l); 
  }

  friend struct std::hash<ManyToOne<L, R>>;
};

template <typename L, typename R>
std::unordered_map<std::unordered_set<L>, R> format_as(ManyToOne<L, R> const &m) {
  std::unordered_map<std::unordered_set<L>, R> result;

  for (R const &r : m.right_values()) {
    result.insert({m.at_r(r), r});
  }

  return result;
}

template <typename L, typename R>
std::ostream &operator<<(std::ostream &s, ManyToOne<L, R> const &m) {
  return (s << fmt::to_string(m));
}

} // namespace FlexFlow

namespace std {

template <typename L, typename R>
struct hash<::FlexFlow::ManyToOne<L, R>> {
  size_t operator()(::FlexFlow::ManyToOne<L, R> const &m) {
    return ::FlexFlow::get_std_hash(m.tie());
  }
};

}

#endif
