#include "kernels/tensor_accessor_reductions.h"
#include "kernels/reduce_tensor_accessor.h"
#include "utils/overload.h"

namespace FlexFlow {

bool tensor_accessor_all(GenericTensorAccessorR const &t) {
  ASSERT(t.data_type == DataType::BOOL);

  return reduce_tensor_accessor_in_all_dims<DataType::BOOL>(
      t,
      overload{
          [](bool lhs, bool rhs) -> bool { return lhs && rhs; },
          [](auto lhs, auto rhs) -> bool { PANIC(); },
      });
}

bool tensor_accessor_any(GenericTensorAccessorR const &t) {
  ASSERT(t.data_type == DataType::BOOL);

  return reduce_tensor_accessor_in_all_dims<DataType::BOOL>(
      t,
      overload{
          [](bool lhs, bool rhs) -> bool { return lhs || rhs; },
          [](auto lhs, auto rhs) -> bool { PANIC(); },
      });
}

} // namespace FlexFlow
