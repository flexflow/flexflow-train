#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIJECTION_BIJECTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIJECTION_BIJECTION_H

#include "utils/bijection/bijection.dtg.h"

namespace FlexFlow {

template <typename L, typename R>
Bijection<R, L> flip_bijection(Bijection<L, R> const &b) {
  return Bijection<R, L>{
      /*l_to_r=*/b.r_to_l,
      /*r_to_l=*/b.l_to_r,
  };
}

} // namespace FlexFlow

#endif
