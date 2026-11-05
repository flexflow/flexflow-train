#include "kernels/reduce_tensor_accessor.h"

namespace FlexFlow {

struct ReduceTensorExampleF {
  template <typename T>
  T operator()(std::vector<T> const &) const {
    PANIC();
  }
};

using F = ReduceTensorExampleF;

template GenericTensorAccessorW
    reduce_tensor_accessor_in_dims(GenericTensorAccessorR const &,
                                   std::set<ff_dim_t> const &,
                                   Allocator &,
                                   F &&);

template int32_t reduce_tensor_accessor_in_all_dims<DataType::INT32>(
    GenericTensorAccessorR const &, F &&);

} // namespace FlexFlow
