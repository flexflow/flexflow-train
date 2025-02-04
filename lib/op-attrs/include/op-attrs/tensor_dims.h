#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_DIMS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_DIMS_H

#include "op-attrs/parallel_tensor_dims.dtg.h"
#include "op-attrs/tensor_dims.dtg.h"

namespace FlexFlow {

FFOrdered<nonnegative_int> const &ff_ordered(TensorDims const &);

nonnegative_int num_dims(TensorDims const &);
nonnegative_int dim_at_idx(TensorDims const &, relative_ff_dim_t);
nonnegative_int &dim_at_idx(TensorDims &, relative_ff_dim_t);

bool tensor_dims_is_broadcastable_to(TensorDims const &curr,
                                     TensorDims const &goal);
std::optional<TensorDims>
    get_broadcast_target_dims(std::unordered_set<TensorDims> const &);

} // namespace FlexFlow

#endif
