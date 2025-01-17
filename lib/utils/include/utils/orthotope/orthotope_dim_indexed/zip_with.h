#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_DIM_INDEXED_ZIP_WITH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_DIM_INDEXED_ZIP_WITH_H

#include "utils/orthotope/orthotope_dim_indexed/orthotope_dim_indexed.h"
#include "utils/containers/intersection.h"

namespace FlexFlow {

template <typename T1, typename T2, typename F, typename Result = std::invoke_result_t<F, T1, T2>>
OrthotopeDimIndexed<Result> zip_with(OrthotopeDimIndexed<T1> const &l, OrthotopeDimIndexed<T2> const &r, F &&f) {
  OrthotopeDimIndexed<Result> result;
  for (orthotope_dim_idx_t i : intersection(l.indices(), r.indices())) {
    result.push_back(f(l.at(i), r.at(i)));
  }

  return result;
}


} // namespace FlexFlow

#endif
