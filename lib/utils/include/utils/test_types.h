#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_TEST_TYPES_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_TEST_TYPES_H

#include "type_traits.h"

namespace FlexFlow {

namespace test_types {

enum capability {
  HASHABLE,
  EQ,
  CMP,
  DEFAULT_CONSTRUCTIBLE,
  COPYABLE,
  PLUS,
  PLUSEQ,
  FMT
};

template <capability PRECONDITION, capability POSTCONDITION>
struct capability_implies : std::false_type {};

template <>
struct capability_implies<CMP, EQ> : std::true_type {};

template <capability C>
struct capability_implies<C, C> : std::true_type {};

template <capability NEEDLE, capability... HAYSTACK>
struct has_capability;

template <capability NEEDLE, capability HEAD, capability... HAYSTACK>
struct has_capability<NEEDLE, HEAD, HAYSTACK...>
    : disjunction<capability_implies<HEAD, NEEDLE>,
                  has_capability<NEEDLE, HAYSTACK...>> {};

template <capability NEEDLE>
struct has_capability<NEEDLE> : std::false_type {};

template <capability... CAPABILITIES>
struct test_type_t {
  template <capability... C>
  using supports = conjunction<has_capability<C, CAPABILITIES...>...>;

  template <capability C = DEFAULT_CONSTRUCTIBLE,
            typename std::enable_if<supports<C>::value, bool>::type = true>
  test_type_t();

  template <capability C = DEFAULT_CONSTRUCTIBLE,
            typename std::enable_if<!supports<C>::value, bool>::type = true>
  test_type_t() = delete;

  template <capability C = COPYABLE,
            typename std::enable_if<supports<C>::value, bool>::type = true>
  test_type_t(test_type_t const &);

  template <capability C = COPYABLE,
            typename std::enable_if<!supports<C>::value, bool>::type = true>
  test_type_t(test_type_t const &) = delete;

  template <capability C = EQ>
  typename std::enable_if<supports<C>::value, bool>::type
      operator==(test_type_t const &) const;

  template <capability C = EQ>
  typename std::enable_if<supports<C>::value, bool>::type
      operator!=(test_type_t const &) const;

  template <capability C = CMP>
  typename std::enable_if<supports<C>::value, bool>::type
      operator<(test_type_t const &) const;

  template <capability C = CMP>
  typename std::enable_if<supports<C>::value, bool>::type
      operator>(test_type_t const &) const;

  template <capability C = CMP>
  typename std::enable_if<supports<C>::value, bool>::type
      operator<=(test_type_t const &) const;

  template <capability C = CMP>
  typename std::enable_if<supports<C>::value, bool>::type
      operator>=(test_type_t const &) const;

  template <capability C = PLUS>
  typename std::enable_if<supports<C>::value, test_type_t>::type
      operator+(test_type_t const &);

  template <capability C = PLUSEQ>
  typename std::enable_if<supports<C>::value, test_type_t>::type
      operator+=(test_type_t const &);
};

template <capability... CAPABILITIES>
enable_if_t<has_capability<FMT, CAPABILITIES...>::value, std::string>
    format_as(test_type_t<CAPABILITIES...>);

using no_eq = test_type_t<>;
using eq = test_type_t<EQ>;
using cmp = test_type_t<CMP>;
using hash_cmp = test_type_t<HASHABLE, CMP>;
using plusable = test_type_t<PLUS, PLUSEQ>;
using fmtable = test_type_t<FMT>;

} // namespace test_types
} // namespace FlexFlow

namespace std {

template <
    ::FlexFlow::test_types::
        capability... CAPABILITIES> //, typename = typename
                                    // std::enable_if<::FlexFlow::test_types::has_capability<::FlexFlow::test_types::HASHABLE>::value,
                                    // bool>::type>
struct hash<::FlexFlow::test_types::test_type_t<CAPABILITIES...>> {
  typename std::enable_if<
      ::FlexFlow::test_types::has_capability<::FlexFlow::test_types::HASHABLE,
                                             CAPABILITIES...>::value,
      size_t>::type
      operator()(
          ::FlexFlow::test_types::test_type_t<CAPABILITIES...> const &) const;
};

} // namespace std

#endif