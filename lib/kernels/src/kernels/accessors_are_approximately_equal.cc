#include "kernels/accessors_are_approximately_equal.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/tensor_accessor_binary_ops.h"
#include "kernels/tensor_accessor_reductions.h"
#include "kernels/values_are_approximately_equal.h"

namespace FlexFlow {

bool accessors_are_approximately_equal(
    GenericTensorAccessorR const &accessor_a,
    GenericTensorAccessorR const &accessor_b,
    std::optional<float> relative_error_allowed) {
  TensorShape shape = require_same(accessor_a.shape, accessor_b.shape);

  ASSERT(shape.data_type == DataType::FLOAT);

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR cpu_a =
      copy_tensor_accessor_r_to_cpu_if_necessary(accessor_a, cpu_allocator);
  GenericTensorAccessorR cpu_b =
      copy_tensor_accessor_r_to_cpu_if_necessary(accessor_b, cpu_allocator);

  for (TensorDimsCoord const &coord : get_tensor_dims_coord_set(shape.dims)) {
    float val_a = cpu_a.at<DataType::FLOAT>(coord);
    float val_b = cpu_b.at<DataType::FLOAT>(coord);
    if (!values_are_approximately_equal(val_a, val_b, relative_error_allowed)) {
      return false;
    }
  }

  return true;
}

} // namespace FlexFlow
