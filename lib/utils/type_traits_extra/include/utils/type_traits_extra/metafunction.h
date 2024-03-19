#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_METAFUNCTION_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_METAFUNCTION_H

#include <type_traits>

namespace FlexFlow {

template <template <typename...> class Func, int N, typename Enable = void>
struct is_nary_metafunction : std::false_type {};

template <template <typename...> class Func, int N>
struct is_nary_metafunction<
    Func,
    N,
    std::enable_if_t<(metafunction_num_args<Func>::value == N)>>
    : std::true_type {};

template <template <typename...> class Func, typename Enable, typename... Args>
struct internal_invoke_metafunction;

template <template <typename...> class Func, typename... Args>
struct internal_invoke_metafunction<
    Func,
    typename std::enable_if<(metafunction_num_args<Func>::value ==
                             (sizeof...(Args)))>::type,
    Args...> : Func<Args...> {};

template <template <typename...> class Func, typename... Args>
using invoke_metafunction = internal_invoke_metafunction<Func, void, Args...>;

} // namespace FlexFlow

#endif