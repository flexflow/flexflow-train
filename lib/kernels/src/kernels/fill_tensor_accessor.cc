#include "kernels/fill_tensor_accessor.h"
#include "kernels/datatype_dispatch.h"
#include "op-attrs/datatype_value.h"

namespace FlexFlow {

template <DataType DT>
struct FillWithZeros {
  void operator()(GenericTensorAccessorW const &accessor) {
    using T = real_type_t<DT>;

    if (accessor.device_type == DeviceType::CPU) {
      memset(accessor.ptr,
             0,
             accessor.shape.num_elements().int_from_positive_int() * sizeof(T));
    } else {
      checkCUDA(cudaMemset(
          accessor.ptr,
          0,
          accessor.shape.num_elements().int_from_positive_int() * sizeof(T)));
    }
  }
};

void fill_with_zeros(GenericTensorAccessorW const &accessor) {
  DataTypeDispatch1<FillWithZeros>{}(accessor.data_type, accessor);
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
