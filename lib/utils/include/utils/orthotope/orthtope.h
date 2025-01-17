#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHTOPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHTOPE_H

namespace FlexFlow {

Orthotope orthotope_from_dim_map(std::unordered_map<orthotope_dim_idx_t, int> const &);

} // namespace FlexFlow

#endif
