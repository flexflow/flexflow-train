#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_TASK_SPACE_DIM_IDX_T_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_TASK_SPACE_DIM_IDX_T_H

#include "op-attrs/operator_task_space_dim_idx_t.dtg.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <set>

namespace FlexFlow {

std::set<operator_task_space_dim_idx_t>
    operator_task_space_dim_idx_range(nonnegative_int end);

} // namespace FlexFlow

#endif
