#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_FILL_TENSOR_ACCESSOR_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_FILL_TENSOR_ACCESSOR_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "op-attrs/datatype_value.dtg.h"

namespace FlexFlow {

void fill_with_zeros(GenericTensorAccessorW const &accessor);

GenericTensorAccessorW create_accessor_w_filled_with(
    TensorShape const &shape, DataTypeValue val, Allocator const &allocator);

GenericTensorAccessorR create_accessor_r_filled_with(
    TensorShape const &shape, DataTypeValue val, Allocator const &allocator);

} // namespace FlexFlow

#endif
