#include "kernels/fill_tensor_accessor.h"
#include "op-attrs/datatype_value.h"

namespace FlexFlow {

void fill_tensor_accessor(GenericTensorAccessorW const &accessor, DataTypeValue val) {
  ASSERT(accessor.device_type == DeviceType::CPU);
  ASSERT(accessor.data_type == get_data_type_of_data_type_value(val));
}

GenericTensorAccessorW create_accessor_w_filled_with(
    TensorShape const &shape, DataTypeValue val, Allocator const &allocator) {
  NOT_IMPLEMENTED();
}

GenericTensorAccessorR create_accessor_r_filled_with(
    TensorShape const &shape, DataTypeValue val, Allocator const &allocator) {
  return read_only_accessor_from_write_accessor(
      create_accessor_w_filled_with(shape, val, allocator));
}

} // namespace FlexFlow
