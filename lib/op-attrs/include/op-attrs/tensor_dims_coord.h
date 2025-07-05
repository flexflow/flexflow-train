#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_DIMS_COORD_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_DIMS_COORD_H

#include "op-attrs/tensor_dims_coord.dtg.h"

namespace FlexFlow {

nonnegative_int tensor_dims_coord_get_num_dims(TensorDimsCoord const &tensor_dims_coord);

TensorDimsCoord tensor_dims_coord_drop_dims(
    TensorDimsCoord const &coord,
    std::function<bool(ff_dim_t)> const &should_drop_dim);

} // namespace FlexFlow

#endif
