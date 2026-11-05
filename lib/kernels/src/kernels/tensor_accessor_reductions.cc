#include "kernels/tensor_accessor_reductions.h"
#include "kernels/reduce_tensor_accessor.h"
#include "utils/containers/all_are_true.h"
#include "utils/containers/any_are_true.h"
#include "utils/containers/get_element_type.h"
#include "utils/overload.h"

namespace FlexFlow {

bool tensor_accessor_all(GenericTensorAccessorR const &t) {
  ASSERT(t.shape.data_type == DataType::BOOL);

  return reduce_tensor_accessor_in_all_dims<DataType::BOOL>(
      t,
      overload{
          [](std::vector<bool> const &bs) -> bool { return all_are_true(bs); },
          [](auto const &x) -> get_element_type_t<decltype(x)> { PANIC(); },
      });
}

bool tensor_accessor_any(GenericTensorAccessorR const &t) {
  ASSERT(t.shape.data_type == DataType::BOOL);

  return reduce_tensor_accessor_in_all_dims<DataType::BOOL>(
      t,
      overload{
          [](std::vector<bool> const &bs) -> bool { return any_are_true(bs); },
          [](auto const &x) -> get_element_type_t<decltype(x)> { PANIC(); },
      });
}

} // namespace FlexFlow
