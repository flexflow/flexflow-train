#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIM_IDX_T_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIM_IDX_T_H

#include "op-attrs/parallel_tensor_dim_idx_t.dtg.h"

namespace FlexFlow {

parallel_tensor_dim_idx_t sum_dim_idx();
parallel_tensor_dim_idx_t discard_copy_dim_idx();
parallel_tensor_dim_idx_t shard_dim_idx(ff_dim_t);

} // namespace FlexFlow

#endif
