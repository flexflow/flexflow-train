#include "kernels/compare_tensor_accessors.h"
#include "kernels/map_tensor_accessors.h"

namespace FlexFlow {

GenericTensorAccessorW compare_tensor_accessors_lt(GenericTensorAccessorR const &lhs,
                                                   GenericTensorAccessorR const &rhs,
                                                   Allocator &output_allocator) {
  return map_tensor_accessors2(lhs, rhs, 
                               DataType::BOOL,
                               [](auto const &l, auto const &r) { return l < r; },
                               output_allocator);
}

GenericTensorAccessorW compare_tensor_accessors_le(GenericTensorAccessorR const &lhs,
                                                   GenericTensorAccessorR const &rhs,
                                                   Allocator &output_allocator) {
  return map_tensor_accessors2(lhs, rhs,
                               DataType::BOOL,
                               [](auto const &l, auto const &r) { return l <= r; },
                               output_allocator);
}


GenericTensorAccessorW compare_tensor_accessors_gt(GenericTensorAccessorR const &lhs,
                                                   GenericTensorAccessorR const &rhs,
                                                   Allocator &output_allocator) {
  return map_tensor_accessors2(lhs, rhs, 
                               DataType::BOOL,
                               [](auto const &l, auto const &r) { return l > r; },
                               output_allocator);
}

GenericTensorAccessorW compare_tensor_accessors_ge(GenericTensorAccessorR const &lhs,
                                                   GenericTensorAccessorR const &rhs,
                                                   Allocator &output_allocator) {
  return map_tensor_accessors2(lhs, rhs, 
                               DataType::BOOL,
                               [](auto const &l, auto const &r) { return l >= r; },
                               output_allocator);
}

GenericTensorAccessorW compare_tensor_accessors_eq(GenericTensorAccessorR const &lhs,
                                                   GenericTensorAccessorR const &rhs,
                                                   Allocator &output_allocator) {
  return map_tensor_accessors2(lhs, rhs, 
                               DataType::BOOL,
                               [](auto const &l, auto const &r) { return l == r; },
                               output_allocator);
}


GenericTensorAccessorW compare_tensor_accessors_ne(GenericTensorAccessorR const &lhs,
                                                   GenericTensorAccessorR const &rhs,
                                                   Allocator &output_allocator) {
  return map_tensor_accessors2(lhs, rhs, 
                               DataType::BOOL,
                               [](auto const &l, auto const &r) { return l != r; },
                               output_allocator);
}

} // namespace FlexFlow
