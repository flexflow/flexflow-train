#include "kernels/linear_kernels_cpu.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/tensor_accessor_binary_ops.h"
#include "kernels/tensor_accessor_unary_ops.h"
#include "utils/exception.h"
#include "utils/nonnegative_int/nonnegative_range.h"

namespace FlexFlow::Kernels::Linear {

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output,
                        GenericTensorAccessorR const &projection,
                        std::optional<GenericTensorAccessorR> const &bias) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  tensor_accessor_matmul_to(input, projection, output);

  if (bias.has_value()) {
    GenericTensorAccessorW broadcasted_bias = tensor_accessor_broadcast(
        bias.value(), 
        tensor_dims_from_array_shape(output.shape),
        cpu_allocator);
    tensor_accessor_elementwise_add_to(
      read_only_accessor_from_write_accessor(output), 
      read_only_accessor_from_write_accessor(broadcasted_bias),
      output);
  }
  
  // for (nonnegative_int i : nonnegative_range(input.shape.at(ff_dim_t{0_n}))) {
  //   for (nonnegative_int j : nonnegative_range(projection.shape.at(ff_dim_t{1_n}))) {
  //     float accum = 0.0f;
  //     if (bias.has_value()) {
  //       accum += bias.value().at<DataType::FLOAT>(FFOrdered{j});
  //     }
  //     for (nonnegative_int k : nonnegative_range(input.shape.at(ff_dim_t{1_n}))) {
  //       accum += input.at<DataType::FLOAT>(FFOrdered{i, k}) * projection.at<DataType::FLOAT>(FFOrdered{k, j});
  //     }
  //     output.at<DataType::FLOAT>(FFOrdered{i, j}) = accum;
  //   }
  // }
}

void cpu_backward_kernel(GenericTensorAccessorR const &output,
                         GenericTensorAccessorR const &output_grad,
                         GenericTensorAccessorR const &input,
                         GenericTensorAccessorW const &input_grad,
                         GenericTensorAccessorR const &projection,
                         GenericTensorAccessorW const &projection_grad,
                         std::optional<GenericTensorAccessorW> const &bias_grad) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  tensor_accessor_matmul_to(output_grad, 
                            read_only_accessor_from_write_accessor(tensor_accessor_transpose(projection, cpu_allocator)),
                            input_grad);
  tensor_accessor_matmul_to(read_only_accessor_from_write_accessor(tensor_accessor_transpose(input, cpu_allocator)),
                            output_grad,
                            projection_grad);

  if (bias_grad.has_value()) {
     
  }
}

} // namespace FlexFlow::Kernels::Linear
