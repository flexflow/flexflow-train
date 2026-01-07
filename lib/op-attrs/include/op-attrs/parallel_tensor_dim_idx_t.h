#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIM_IDX_T_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIM_IDX_T_H

#include "op-attrs/num_ptensor_shard_dims_t.dtg.h"
#include "op-attrs/parallel_tensor_dim_idx_t.dtg.h"
#include "utils/orthotope/dim_ordering.dtg.h"

namespace FlexFlow {

parallel_tensor_dim_idx_t sum_dim_idx();
parallel_tensor_dim_idx_t discard_copy_dim_idx();
parallel_tensor_dim_idx_t shard_dim_idx(ff_dim_t);

bool is_dim_idx_for_reduction_dimension(parallel_tensor_dim_idx_t);

std::set<parallel_tensor_dim_idx_t>
    dim_idxs_for_num_shard_dims(num_ptensor_shard_dims_t num_shard_dims);

DimOrdering<parallel_tensor_dim_idx_t> get_parallel_tensor_dim_ordering();

} // namespace FlexFlow

#endif
