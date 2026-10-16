#include "kernels/reduce_kernels_cpu.h"
#include "kernels/fill_tensor_accessor.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/reduce_tensor_accessor.h"
#include "kernels/tensor_accessor_promote_dims.h"
#include "kernels/tensor_accessor_unary_ops.h"
#include "op-attrs/datatype_value.h"
#include "utils/containers/compute_mean.h"
#include "utils/containers/sum.h"
#include "utils/overload.h"

namespace FlexFlow {

void reduce_cpu_forward_kernel(ReduceAttrs const &attrs,
                               GenericTensorAccessorR const &input,
                               GenericTensorAccessorW const &output) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  auto reduce_values = [&](std::vector<float> const &input_values) -> float {
    switch (attrs.op) {
      case ReduceOp::SUM:
        return sum(input_values);
      case ReduceOp::AVG:
        return compute_mean(input_values);
      default:
        PANIC(fmt::format("Unknown ReduceOp {}", attrs.op));
    };
  };

  GenericTensorAccessorW result = reduce_tensor_accessor_in_dims(
      input,
      attrs.axes.unwrap_as_set(),
      cpu_allocator,
      overload{
          [&](std::vector<float> const &input_values) -> float {
            return reduce_values(input_values);
          },
          [](auto const &x) -> get_element_type_t<decltype(x)> { PANIC(); },
      });

  if (attrs.keepdims) {
    TensorDims promoted_dims = [&]() {
      TensorDims promoted = input.shape.dims;
      for (ff_dim_t d : attrs.axes) {
        promoted.ff_ordered.at(d) = 1_p;
      }
      return promoted;
    }();

    result = tensor_accessor_broadcast(
        tensor_accessor_promote_dims(result, promoted_dims),
        input.shape.dims,
        cpu_allocator);
  }

  copy_accessor_data_to_l_from_r_strict(output, result);
}

void reduce_cpu_backward_kernel(ReduceAttrs const &attrs,
                                GenericTensorAccessorR const &output_grad,
                                GenericTensorAccessorW const &input_grad) {
  float value = [&]() -> float {
    switch (attrs.op) {
      case ReduceOp::SUM:
        return 1.0;
      case ReduceOp::AVG: {
        nonnegative_int reduction_size =
            get_num_elements(input_grad.shape.dims) /
            get_num_elements(output_grad.shape.dims);
        return 1.0f / reduction_size;
      }
      default:
        PANIC(fmt::format("Unknown ReduceOp {}", attrs.op));
    };
  }();

  fill_accessor_with_constant(input_grad, make_float_data_type_value(value));
}

} // namespace FlexFlow
