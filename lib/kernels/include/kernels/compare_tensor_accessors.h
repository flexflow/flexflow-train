#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_COMPARE_TENSOR_ACCESSORS_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_COMPARE_TENSOR_ACCESSORS_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"

namespace FlexFlow {

GenericTensorAccessorW compare_tensor_accessors_lt(GenericTensorAccessorR const &lhs,
                                                   GenericTensorAccessorR const &rhs,
                                                   Allocator &allocator);

GenericTensorAccessorW compare_tensor_accessors_le(GenericTensorAccessorR const &lhs,
                                                   GenericTensorAccessorR const &rhs,
                                                   Allocator &allocator);

GenericTensorAccessorW compare_tensor_accessors_gt(GenericTensorAccessorR const &lhs,
                                                   GenericTensorAccessorR const &rhs,
                                                   Allocator &allocator);

GenericTensorAccessorW compare_tensor_accessors_ge(GenericTensorAccessorR const &lhs,
                                                   GenericTensorAccessorR const &rhs,
                                                   Allocator &allocator);

GenericTensorAccessorW compare_tensor_accessors_eq(GenericTensorAccessorR const &lhs,
                                                   GenericTensorAccessorR const &rhs,
                                                   Allocator &allocator);

GenericTensorAccessorW compare_tensor_accessors_ne(GenericTensorAccessorR const &lhs,
                                                   GenericTensorAccessorR const &rhs,
                                                   Allocator &allocator);

} // namespace FlexFlow

#endif
