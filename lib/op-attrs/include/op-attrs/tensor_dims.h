#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_DIMS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_DIMS_H

#include "op-attrs/parallel_tensor_dims.dtg.h"
#include "op-attrs/tensor_dims.dtg.h"
#include "op-attrs/tensor_dims_coord.dtg.h"
#include "utils/bidict/bidict.h"

namespace FlexFlow {

FFOrdered<positive_int> const &ff_ordered(TensorDims const &);

bool tensor_dims_has_dim(TensorDims const &, ff_dim_t);

nonnegative_int get_num_dims(TensorDims const &);

positive_int dim_at_idx(TensorDims const &, relative_ff_dim_t);
positive_int &dim_at_idx(TensorDims &, relative_ff_dim_t);

positive_int dim_at_idx(TensorDims const &, ff_dim_t);
positive_int &dim_at_idx(TensorDims &, ff_dim_t);

std::optional<positive_int> try_dim_at_idx(TensorDims const &,
                                           relative_ff_dim_t);
std::optional<positive_int> try_dim_at_idx(TensorDims const &, ff_dim_t);

positive_int get_num_elements(TensorDims const &);

bool tensor_dims_is_broadcastable_to(TensorDims const &curr,
                                     TensorDims const &goal);

bool tensor_dims_contains_coord(TensorDims const &tensor_dims,
                                TensorDimsCoord const &coord);

TensorDimsCoord get_broadcast_src_coord(TensorDims const &input_dims,
                                        TensorDims const &output_dims,
                                        TensorDimsCoord const &dst_coord);

std::unordered_set<TensorDimsCoord>
    get_tensor_dims_coord_set(TensorDims const &tensor_dims);

std::unordered_set<ff_dim_t> get_ff_dim_t_set(TensorDims const &);

std::optional<TensorDims>
    get_broadcast_target_dims(std::unordered_set<TensorDims> const &);

TensorDims
    tensor_dims_drop_dims(TensorDims const &dims,
                          std::function<bool(ff_dim_t)> const &should_drop_dim);

TensorDims slice_tensor_dims(TensorDims const &,
                             ff_dim_t const &start,
                             std::optional<ff_dim_t> const &stop);

TensorDims slice_tensor_dims(TensorDims const &,
                             relative_ff_dim_t const &start,
                             std::optional<relative_ff_dim_t> const &stop);

} // namespace FlexFlow

#endif
