#include "kernels/map_tensor_accessors.h"

namespace FlexFlow {

struct F1 {
  template <typename T>
  float operator()(T const &t) const { NOT_IMPLEMENTED(); }
};

template
GenericTensorAccessorW map_tensor_accessor(GenericTensorAccessorR const &,
                                           Allocator &,
                                           F1 &&);

struct F2 {
  template <typename T>
  float operator()(T const &lhs, T const &rhs) const { NOT_IMPLEMENTED(); }
};

template
  GenericTensorAccessorW map_tensor_accessors2(GenericTensorAccessorR const &,
                                               GenericTensorAccessorR const &,
                                               Allocator &,
                                               F2 &&);

} // namespace FlexFlow
