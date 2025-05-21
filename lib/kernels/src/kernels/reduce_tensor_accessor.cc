#include "kernels/reduce_tensor_accessor.h"

namespace FlexFlow {

using F = std::function<int32_t(int32_t, float)>;

template GenericTensorAccessorW
    reduce_tensor_accessor_in_dims(GenericTensorAccessorR const &,
                                   std::unordered_set<ff_dim_t> const &,
                                   Allocator &,
                                   F &&);

template int32_t reduce_tensor_accessor_in_all_dims<DataType::INT32>(
    GenericTensorAccessorR const &, F &&);

} // namespace FlexFlow
