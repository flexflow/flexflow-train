#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_DIM_INDEXED_ORTHOTOPE_DIM_INDEXED_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_DIM_INDEXED_ORTHOTOPE_DIM_INDEXED_OF_H

#include "utils/orthotope/orthotope_dim_indexed/orthotope_dim_indexed.h"
#include <vector>

namespace FlexFlow {

template <typename T>
OrthotopeDimIndexed<T> orthotope_dim_indexed_of(std::vector<T> const &v) {
  return OrthotopeDimIndexed<T>(v.cbegin(), v.cend());
}

} // namespace FlexFlow

#endif
