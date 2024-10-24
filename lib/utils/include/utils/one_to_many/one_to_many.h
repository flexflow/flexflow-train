#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_H

#include <unordered_set>
#include <unordered_map>
#include "utils/containers/try_at.h"
#include <fmt/format.h>
#include "utils/hash-utils.h"
#include "utils/hash/unordered_map.h"
#include "utils/hash/unordered_set.h"
#include "utils/containers/keys.h"
#include "utils/exception.h"

namespace FlexFlow {

template <typename L, typename R>
struct OneToMany {
public:
  OneToMany()
    : l_to_r(), r_to_l()
  { }

  bool operator==(OneToMany const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(OneToMany const &other) const {
    return this->tie() != other.tie();
  }

  void insert(std::pair<L, R> const &p) {
    L l = p.first;
    R r = p.second;

    std::optional<L> found_l = try_at(this->r_to_l, r);

    if (!found_l.has_value()) {
      this->r_to_l.insert({r, l});
      this->l_to_r[l].insert(r);
    } else if (found_l.value() == l) {
      return;
    } else {
      throw mk_runtime_error(fmt::format("Existing mapping found for right value {}: tried to map to left value {}, but is already bound to left value {}", r, l, found_l.value()));
    }
  }
  
  std::unordered_set<R> const &at_l(L const &l) const {
    return this->l_to_r.at(l);
  }

  L const &at_r(R const &r) const {
    return this->r_to_l.at(r);
  }

  std::unordered_set<L> left_values() const {
    return keys(this->l_to_r);
  }

  std::unordered_set<R> right_values() const {
    return keys(this->r_to_l);
  }
private:
  std::unordered_map<L, std::unordered_set<R>> l_to_r;
  std::unordered_map<R, L> r_to_l;
private:
  std::tuple<decltype(l_to_r) const &, decltype(r_to_l) const &> tie() const {
    return std::tie(this->l_to_r, this->r_to_l); 
  }

  friend struct std::hash<OneToMany<L, R>>;
};

} // namespace FlexFlow

namespace std {

template <typename L, typename R>
struct hash<::FlexFlow::OneToMany<L, R>> {
  size_t operator()(::FlexFlow::OneToMany<L, R> const &m) {
    return ::FlexFlow::get_std_hash(m.tie());
  }
};

}

#endif
