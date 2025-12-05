#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_SINGULAR_OR_VARIADIC_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_SINGULAR_OR_VARIADIC_H

#include "utils/singular_or_variadic.dtg.h"
#include "utils/overload.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip_strict.h"

namespace FlexFlow {

template <typename T, typename F, typename Out = std::invoke_result_t<F, T>>
SingularOrVariadic<Out> transform_singular_or_variadic(SingularOrVariadic<T> const &sv, F &&f) {
  return sv.template visit<
    SingularOrVariadic<Out>
  >(overload {
    [&](T const &singular) -> SingularOrVariadic<Out> {
      return SingularOrVariadic<Out>{
        f(singular),
      };
    },
    [&](std::vector<T> const &variadic) -> SingularOrVariadic<Out> {
      return SingularOrVariadic<Out>{
        transform(variadic, f),
      };
    }
  });
}

template <typename T1, typename T2>
SingularOrVariadic<std::pair<T1, T2>> zip_strict_singular_or_variadic(
    SingularOrVariadic<T1> const &sv1,
    SingularOrVariadic<T2> const &sv2) 
{
  if (sv1.is_singular() && sv2.is_singular()) {
    return SingularOrVariadic<std::pair<T1, T2>>{
      std::pair{
        sv1.require_singular(),
        sv2.require_singular(),
      },
    };
  } else {
    ASSERT(sv1.is_variadic() && sv2.is_variadic());

    return SingularOrVariadic<std::pair<T1, T2>>{
      zip_strict(sv1.require_variadic(), sv2.require_variadic()),
    };
  }
}

template <typename T1, 
          typename T2, 
          typename F,
          typename Out = std::invoke_result_t<F, T1, T2>>
SingularOrVariadic<Out> zip_strict_singular_or_variadic_with(
    SingularOrVariadic<T1> const &sv1,
    SingularOrVariadic<T2> const &sv2,
    F &&f) 
{
  return transform_singular_or_variadic(
    zip_strict_singular_or_variadic(
      sv1,
      sv2),
    [&](std::pair<T1, T2> const &p) -> Out {
      return f(p.first, p.second);
    });
}


} // namespace FlexFlow

#endif
