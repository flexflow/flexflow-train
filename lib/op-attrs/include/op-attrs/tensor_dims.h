#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_DIMS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_DIMS_H

#include "op-attrs/parallel_tensor_dims.dtg.h"
#include "op-attrs/tensor_dims.dtg.h"

namespace FlexFlow {

FFOrdered<nonnegative_int> const &ff_ordered(TensorDims const &);

nonnegative_int num_dims(TensorDims const &);
nonnegative_int dim_at_idx(TensorDims const &, relative_ff_dim_t);
nonnegative_int &dim_at_idx(TensorDims &, relative_ff_dim_t);
nonnegative_int get_num_elements(TensorDims const &);

bool tensor_dims_is_broadcastable_to(TensorDims const &curr,
                                     TensorDims const &goal);
std::optional<TensorDims>
    get_broadcast_target_dims(std::unordered_set<TensorDims> const &);

TensorDims slice_tensor_dims(TensorDims const &,
                             std::optional<relative_ff_dim_t> const &start,
                             std::optional<relative_ff_dim_t> const &stop);

} // namespace FlexFlow

#endif
