#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_DIM_INDEXED_DROP_IDXS_EXCEPT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_DIM_INDEXED_DROP_IDXS_EXCEPT_H

#include "utils/containers/contains.h"
#include "utils/orthotope/orthotope_dim_indexed/orthotope_dim_indexed.h"
#include "utils/exception.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/fmt/set.h"

namespace FlexFlow {

template <typename T>
OrthotopeDimIndexed<T> drop_idxs_except(OrthotopeDimIndexed<T> const &d, std::set<orthotope_dim_idx_t> const &keep) {
  OrthotopeDimIndexed<T> result;

  if (!is_subseteq_of(d.indices(), keep)) {
    throw mk_runtime_error(fmt::format("drop_idxs_except expected keep to be a subset of d's dims, but got d={}, keep={}", d, keep));
  }

  for (orthotope_dim_idx_t const &idx : d.indices()) {
    if (contains(keep, idx)) {
      result.push_back(d.at(idx));
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
