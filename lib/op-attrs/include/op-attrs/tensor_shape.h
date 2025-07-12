#ifndef _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H
#define _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H

#include "op-attrs/tensor_shape.dtg.h"
#include "utils/units/num_bytes_t.h"

namespace FlexFlow {

num_bytes_t get_size_in_bytes(TensorShape const &);

TensorShape tensor_shape_drop_dims(
    TensorShape const &coord,
    std::function<bool(ff_dim_t)> const &should_drop_dim);

TensorShape slice_tensor_shape(TensorShape const &,
                               relative_ff_dim_t const &start,
                               std::optional<relative_ff_dim_t> const &stop);

} // namespace FlexFlow

#endif
