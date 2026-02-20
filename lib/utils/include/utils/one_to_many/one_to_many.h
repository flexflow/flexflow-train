#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_H

#include "utils/containers/generate_map.h"
#include "utils/containers/keys.h"
#include "utils/containers/try_at.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/values.h"
#include "utils/exception.h"
#include "utils/fmt/unordered_map.h"
#include "utils/fmt/unordered_set.h"
#include "utils/hash-utils.h"
#include "utils/hash/tuple.h"
#include "utils/hash/unordered_map.h"
#include "utils/hash/unordered_set.h"
#include "utils/json/check_is_json_deserializable.h"
#include "utils/json/check_is_json_serializable.h"
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <rapidcheck.h>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

template <typename L, typename R>
struct OneToMany {
public:
  OneToMany() : m_l_to_r(), m_r_to_l() {}

  template <typename It>
  OneToMany(It start, It end) : OneToMany() {
    for (; start < end; start++) {
      ASSERT(start->second.size() > 0);
      for (R const &r : start->second) {
        this->insert(std::pair<L, R>{start->first, r});
      }
    }
  }

  OneToMany(std::initializer_list<std::pair<L, std::initializer_list<R>>> const
                &l_to_r)
      : OneToMany(l_to_r.begin(), l_to_r.end()) {}

  bool operator==(OneToMany const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(OneToMany const &other) const {
    return this->tie() != other.tie();
  }

  void insert(std::pair<L, R> const &p) {
    L l = p.first;
    R r = p.second;

    std::optional<L> found_l = try_at(this->m_r_to_l, r);

    if (!found_l.has_value()) {
      this->m_r_to_l.insert({r, l});
      this->m_l_to_r[l].insert(r);
    } else if (found_l.value() == l) {
      return;
    } else {
      throw mk_runtime_error(
          fmt::format("Existing mapping found for right value {}: tried to map "
                      "to left value {}, but is already bound to left value {}",
                      r,
                      l,
                      found_l.value()));
    }
  }

  std::unordered_set<R> const &at_l(L const &l) const {
    return this->m_l_to_r.at(l);
  }

  L const &at_r(R const &r) const {
    return this->m_r_to_l.at(r);
  }

  std::unordered_set<L> left_values() const {
    return keys(this->m_l_to_r);
  }

  std::unordered_set<R> right_values() const {
    return keys(this->m_r_to_l);
  }

  std::unordered_set<std::unordered_set<R>> right_groups() const {
    return unordered_set_of(values(this->m_l_to_r));
  }

  std::unordered_map<L, std::unordered_set<R>> const &l_to_r() const {
    return this->m_l_to_r;
  }

  std::unordered_map<R, L> const &r_to_l() const {
    return this->m_r_to_l;
  }

private:
  std::unordered_map<L, std::unordered_set<R>> m_l_to_r;
  std::unordered_map<R, L> m_r_to_l;

private:
  std::tuple<decltype(m_l_to_r) const &, decltype(m_r_to_l) const &>
      tie() const {
    return std::tie(this->m_l_to_r, this->m_r_to_l);
  }

  friend struct std::hash<OneToMany<L, R>>;
};

template <typename L, typename R>
std::unordered_map<L, std::unordered_set<R>>
    format_as(OneToMany<L, R> const &m) {
  return generate_map(m.left_values(), [&](L const &l) { return m.at_l(l); });
}

template <typename L, typename R>
std::ostream &operator<<(std::ostream &s, OneToMany<L, R> const &m) {
  return (s << fmt::to_string(m));
}

} // namespace FlexFlow

namespace nlohmann {

template <typename L, typename R>
struct adl_serializer<::FlexFlow::OneToMany<L, R>> {
  static ::FlexFlow::OneToMany<L, R> from_json(json const &j) {
    CHECK_IS_JSON_DESERIALIZABLE(L);
    CHECK_IS_JSON_DESERIALIZABLE(R);

    NOT_IMPLEMENTED();
  }

  static void to_json(json &j, ::FlexFlow::OneToMany<L, R> const &m) {
    CHECK_IS_JSON_SERIALIZABLE(L);
    CHECK_IS_JSON_SERIALIZABLE(R);

    NOT_IMPLEMENTED();
  }
};

} // namespace nlohmann

namespace rc {

template <typename L, typename R>
struct Arbitrary<::FlexFlow::OneToMany<L, R>> {
  static Gen<::FlexFlow::OneToMany<L, R>> arbitrary() {
    NOT_IMPLEMENTED();
  }
};

} // namespace rc

namespace std {

template <typename L, typename R>
struct hash<::FlexFlow::OneToMany<L, R>> {
  size_t operator()(::FlexFlow::OneToMany<L, R> const &m) {
    return ::FlexFlow::get_std_hash(m.tie());
  }
};

} // namespace std

#endif
