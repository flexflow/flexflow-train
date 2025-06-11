#ifndef _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H
#define _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H

#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

nonnegative_int num_dims(TensorShape const &);
positive_int dim_at_idx(TensorShape const &, relative_ff_dim_t);
positive_int &dim_at_idx(TensorShape &, relative_ff_dim_t);
positive_int get_num_elements(TensorShape const &);
positive_int get_size_in_bytes(TensorShape const &);

TensorShape slice_tensor_shape(TensorShape const &,
                               relative_ff_dim_t const &start,
                               std::optional<relative_ff_dim_t> const &stop);

} // namespace FlexFlow

#endif
