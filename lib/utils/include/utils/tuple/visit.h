#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_TUPLE_VISIT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_TUPLE_VISIT_H

#include <utility>
#include <tuple>

namespace FlexFlow {

template <typename Tuple, typename Visitor, std::size_t... Idxs>
void visit_tuple_impl(Tuple const &tuple, Visitor &&v, std::index_sequence<Idxs...>) {
  (v(std::get<Idxs>(tuple)), ...);
}

template <typename Visitor, typename... Ts>
void visit_tuple(std::tuple<Ts...> const &tuple, Visitor &&v) {
  visit_tuple_impl(tuple, v, std::index_sequence_for<Ts...>{});
}

} // namespace FlexFlow

#endif
