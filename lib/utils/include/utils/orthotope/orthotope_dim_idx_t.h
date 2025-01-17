#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_DIM_IDX_T_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_DIM_IDX_T_H

#include "utils/orthotope/orthotope_dim_idx_t.dtg.h"
#include <set>

namespace FlexFlow {

std::set<orthotope_dim_idx_t> dim_idxs_for_orthotope_with_num_dims(int num_dims);

} // namespace FlexFlow

#endif
