#include "kernels/map_tensor_accessors.h"

namespace FlexFlow {

struct F1 {
  template <typename T>
  float operator()(T const &t) const {
    NOT_IMPLEMENTED();
  }
};

template GenericTensorAccessorW
    map_tensor_accessor(GenericTensorAccessorR const &, F1 &&, Allocator &);

struct F2 {
  template <typename T1, typename T2>
  float operator()(T1 const &lhs, T2 const &rhs) const {
    NOT_IMPLEMENTED();
  }
};

template GenericTensorAccessorW
    map_tensor_accessors2(GenericTensorAccessorR const &,
                          GenericTensorAccessorR const &,
                          DataType,
                          F2 &&,
                          Allocator &);

} // namespace FlexFlow
