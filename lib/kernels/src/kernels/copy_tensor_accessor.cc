#include "kernels/copy_tensor_accessor.h"
#include "kernels/datatype_dispatch.h"

namespace FlexFlow {

template <DataType DT>
struct CopyTensorAccessorW {
  GenericTensorAccessorW operator()(GenericTensorAccessorW const &src_accessor,
                                    Allocator &allocator) {
    TensorShape shape =
        get_tensor_shape(src_accessor.shape, src_accessor.data_type);
    GenericTensorAccessorW dst_accessor = allocator.allocate_tensor(shape);

    copy_accessor_data_to_l_from_r(dst_accessor, src_accessor);

    return dst_accessor;
  }
};

GenericTensorAccessorW
    copy_tensor_accessor_w(GenericTensorAccessorW const &src_accessor,
                           Allocator &allocator) {
  return DataTypeDispatch1<CopyTensorAccessorW>{}(
      src_accessor.data_type, src_accessor, allocator);
}

template <DataType DT>
struct CopyTensorAccessorR {
  GenericTensorAccessorR operator()(GenericTensorAccessorR const &src_accessor,
                                    Allocator &allocator) {
    TensorShape shape =
        get_tensor_shape(src_accessor.shape, src_accessor.data_type);
    GenericTensorAccessorW dst_accessor = allocator.allocate_tensor(shape);

    copy_accessor_data_to_l_from_r(dst_accessor, src_accessor);

    return read_only_accessor_from_write_accessor(dst_accessor);
  }
};

GenericTensorAccessorR
    copy_tensor_accessor_r(GenericTensorAccessorR const &src_accessor,
                           Allocator &allocator) {
  return DataTypeDispatch1<CopyTensorAccessorR>{}(
      src_accessor.data_type, src_accessor, allocator);
}

GenericTensorAccessorR copy_tensor_accessor_r_to_cpu_if_necessary(
    GenericTensorAccessorR const &accessor, Allocator &cpu_allocator) {
  if (accessor.device_type == DeviceType::GPU) {
    return copy_tensor_accessor_r(accessor, cpu_allocator);
  } else {
    return accessor;
  }
}

GenericTensorAccessorW copy_tensor_accessor_w_to_cpu_if_necessary(
    GenericTensorAccessorW const &accessor, Allocator &cpu_allocator) {
  if (accessor.device_type == DeviceType::GPU) {
    return copy_tensor_accessor_w(accessor, cpu_allocator);
  } else {
    return accessor;
  }
}

} // namespace FlexFlow
