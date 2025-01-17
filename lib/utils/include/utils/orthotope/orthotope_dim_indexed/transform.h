#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_DIM_INDEXED_TRANSFORM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_DIM_INDEXED_TRANSFORM_H

#include "utils/orthotope/orthotope_dim_indexed/orthotope_dim_indexed.h"
#include "utils/orthotope/orthotope_dim_indexed/orthotope_dim_indexed_of.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

template <typename T, typename F, typename Result = std::invoke_result_t<F, T>>
OrthotopeDimIndexed<Result> transform(OrthotopeDimIndexed<T> const &d, F &&f) {
  return orthotope_dim_indexed_of(transform(d.get_contents(), f));
}

} // namespace FlexFlow

#endif
