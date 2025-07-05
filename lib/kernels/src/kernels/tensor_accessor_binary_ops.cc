#include "kernels/tensor_accessor_binary_ops.h"
#include "kernels/map_tensor_accessors.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

GenericTensorAccessorW
    tensor_accessor_elementwise_add(GenericTensorAccessorR const &lhs,
                                    GenericTensorAccessorR const &rhs,
                                    Allocator &output_allocator) {
  return map_tensor_accessors2(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return l + r; },
      output_allocator);
}

void tensor_accessor_elementwise_add_to(GenericTensorAccessorR const &lhs,
                                        GenericTensorAccessorR const &rhs,
                                        GenericTensorAccessorW const &output) {
  map_tensor_accessors2_to(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return l + r; },
      output);
}

GenericTensorAccessorW
    tensor_accessor_elementwise_subtract(GenericTensorAccessorR const &lhs,
                                         GenericTensorAccessorR const &rhs,
                                         Allocator &output_allocator) {
  return map_tensor_accessors2(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return l - r; },
      output_allocator);
}

void tensor_accessor_elementwise_subtract_to(
    GenericTensorAccessorR const &lhs,
    GenericTensorAccessorR const &rhs,
    GenericTensorAccessorW const &output) {
  map_tensor_accessors2_to(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return l - r; },
      output);
}

GenericTensorAccessorW
    tensor_accessor_elementwise_multiply(GenericTensorAccessorR const &lhs,
                                         GenericTensorAccessorR const &rhs,
                                         Allocator &output_allocator) {
  return map_tensor_accessors2(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return l * r; },
      output_allocator);
}

void tensor_accessor_elementwise_multiply_to(
    GenericTensorAccessorR const &lhs,
    GenericTensorAccessorR const &rhs,
    GenericTensorAccessorW const &output) {
  map_tensor_accessors2_to(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return l * r; },
      output);
}

static TensorShape get_matmul_output_shape(TensorShape const &lhs,
                                           TensorShape const &rhs) {
  ASSERT(get_num_dims(lhs.dims) == 2);
  ASSERT(get_num_dims(rhs.dims) == 2);
  ASSERT(lhs.data_type == DataType::FLOAT);
  ASSERT(rhs.data_type == DataType::FLOAT);
  ASSERT(dim_at_idx(lhs.dims, relative_ff_dim_t{1}) ==
         dim_at_idx(rhs.dims, relative_ff_dim_t{0}));

  return TensorShape{
      TensorDims{FFOrdered{
          dim_at_idx(lhs.dims, relative_ff_dim_t{0}),
          dim_at_idx(rhs.dims, relative_ff_dim_t{1}),
      }},
      DataType::FLOAT,
  };
}

GenericTensorAccessorW tensor_accessor_matmul(GenericTensorAccessorR const &lhs,
                                              GenericTensorAccessorR const &rhs,
                                              Allocator &output_allocator) {
  TensorShape output_shape =
      get_matmul_output_shape(get_tensor_shape_for_accessor_r(lhs),
                              get_tensor_shape_for_accessor_r(rhs));

  GenericTensorAccessorW output =
      output_allocator.allocate_tensor(output_shape);

  tensor_accessor_matmul_to(lhs, rhs, output);

  return output;
}

void tensor_accessor_matmul_to(GenericTensorAccessorR const &lhs,
                               GenericTensorAccessorR const &rhs,
                               GenericTensorAccessorW const &output) {
  TensorShape output_shape =
      get_matmul_output_shape(get_tensor_shape_for_accessor_r(lhs),
                              get_tensor_shape_for_accessor_r(rhs));

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR lhs_cpu =
      copy_tensor_accessor_r_to_cpu_if_necessary(lhs, cpu_allocator);
  GenericTensorAccessorR rhs_cpu =
      copy_tensor_accessor_r_to_cpu_if_necessary(rhs, cpu_allocator);
  GenericTensorAccessorW output_cpu =
      cpu_allocator.allocate_tensor(output_shape);

  for (nonnegative_int i : nonnegative_range(dim_at_idx(lhs.shape.dims, ff_dim_t{0_n}))) {
    for (nonnegative_int j : nonnegative_range(dim_at_idx(rhs.shape.dims, ff_dim_t{1_n}))) {
      float accum = 0.0f;
      for (nonnegative_int k : nonnegative_range(dim_at_idx(lhs.shape.dims, ff_dim_t{1_n}))) {
        accum += lhs_cpu.at<DataType::FLOAT>(TensorDimsCoord{FFOrdered{i, k}}) *
                 rhs_cpu.at<DataType::FLOAT>(TensorDimsCoord{FFOrdered{k, j}});
      }
      output_cpu.at<DataType::FLOAT>(TensorDimsCoord{FFOrdered{i, j}}) = accum;
    }
  }

  return copy_accessor_data_to_l_from_r(output, output_cpu);
}

} // namespace FlexFlow
