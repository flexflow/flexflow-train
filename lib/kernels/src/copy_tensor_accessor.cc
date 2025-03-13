#include "kernels/copy_tensor_accessor.h"
#include "kernels/datatype_dispatch.h"

namespace FlexFlow {

void copy_accessor_data_to_l_from_r(
    GenericTensorAccessorW &dst_accessor,
    GenericTensorAccessorR const &src_accessor) {
  size_t num_bytes =
      dst_accessor.shape.get_volume().unwrap_nonnegative() *
      size_of_datatype(dst_accessor.data_type).unwrap_nonnegative();

  DeviceType dst_device_type = dst_accessor.device_type;
  DeviceType src_device_type = src_accessor.device_type;

  if (src_device_type == DeviceType::CPU &&
      dst_device_type == DeviceType::CPU) {
    memcpy(dst_accessor.ptr, src_accessor.ptr, num_bytes);
  } else if (src_device_type == DeviceType::CPU &&
             dst_device_type == DeviceType::GPU) {
    checkCUDA(cudaMemcpy(
        dst_accessor.ptr, src_accessor.ptr, num_bytes, cudaMemcpyHostToDevice));
  } else if (src_device_type == DeviceType::GPU &&
             dst_device_type == DeviceType::CPU) {
    checkCUDA(cudaMemcpy(
        dst_accessor.ptr, src_accessor.ptr, num_bytes, cudaMemcpyDeviceToHost));
  } else {
    assert(src_device_type == DeviceType::GPU);
    assert(dst_device_type == DeviceType::GPU);
    checkCUDA(cudaMemcpy(dst_accessor.ptr,
                         src_accessor.ptr,
                         num_bytes,
                         cudaMemcpyDeviceToDevice));
  }
}

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

GenericTensorAccessorR
    copy_accessor_r_to_cpu_if_necessary(GenericTensorAccessorR const &accessor,
                                        Allocator &cpu_allocator) {
  if (cpu_allocator.get_allocation_device_type() == DeviceType::GPU) {
    throw mk_runtime_error("Allocator must be a CPU allocator");
  }

  GenericTensorAccessorR cpu_accessor = accessor;
  if (accessor.device_type == DeviceType::GPU) {
    cpu_accessor = copy_tensor_accessor_r(accessor, cpu_allocator);
  }
  return cpu_accessor;
}

GenericTensorAccessorW
    copy_accessor_w_to_cpu_if_necessary(GenericTensorAccessorW const &accessor,
                                        Allocator &cpu_allocator) {
  if (cpu_allocator.get_allocation_device_type() == DeviceType::GPU) {
    throw mk_runtime_error("Allocator must be a CPU allocator");
  }

  GenericTensorAccessorW cpu_accessor = accessor;
  if (accessor.device_type == DeviceType::GPU) {
    cpu_accessor = copy_tensor_accessor_w(accessor, cpu_allocator);
  }
  return cpu_accessor;
}

} // namespace FlexFlow
