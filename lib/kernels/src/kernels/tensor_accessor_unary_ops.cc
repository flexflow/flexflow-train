#include "kernels/tensor_accessor_unary_ops.h"
#include "kernels/array_coord.h"
#include "kernels/map_tensor_accessors.h"
#include "op-attrs/datatype_value.h"
#include "kernels/datatype_dispatch.h"
#include "op-attrs/ff_ordered/slice.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/ff_ordered/reversed.h"
#include "op-attrs/ff_ordered/concat.h"
#include "kernels/fill_tensor_accessor.h"

namespace FlexFlow {

GenericTensorAccessorW
  tensor_accessor_scale_by_constant(GenericTensorAccessorR const &t,
                                    float constant,
                                    Allocator &output_allocator) {
  ASSERT(t.data_type == DataType::FLOAT);

  return map_tensor_accessor(t,
                             [&](auto const &elem) {
                               return elem * constant;
                             },
                             output_allocator);
}

void
  tensor_accessor_scale_by_constant_inplace(GenericTensorAccessorW const &t,
                                            float constant) {
  ASSERT(t.data_type == DataType::FLOAT);

  return map_tensor_accessor_inplace(t,
                                     [&](auto const &elem) {
                                       return elem * constant;
                                     });
}

template <DataType DT>
struct CPUTensorAccessorBroadcast {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    TensorDims input_dims = tensor_dims_from_array_shape(input.shape);
    TensorDims output_dims = tensor_dims_from_array_shape(output.shape);

    for (ArrayCoord const &output_coord : get_array_coord_set(output.shape)) {
      TensorDimsCoord output_dims_coord = tensor_dims_coord_from_array_coord(output_coord);

      TensorDimsCoord input_dims_coord = get_broadcast_src_coord(
        /*input_dims=*/input_dims,
        /*output_dims=*/output_dims,
        /*dst_coord=*/output_dims_coord);

      output.at<DT>(output_dims_coord.ff_ordered) = input.at<DT>(input_dims_coord.ff_ordered);
    }
  }
};

void tensor_accessor_broadcast_to(GenericTensorAccessorR const &input,
                                  TensorDims const &output_dims,
                                  GenericTensorAccessorW const &output) {
  TensorDims input_dims = tensor_dims_from_array_shape(input.shape);
  ASSERT(tensor_dims_is_broadcastable_to(input_dims, output_dims));

  TensorShape output_shape = TensorShape{output_dims, input.data_type};

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR input_cpu = copy_tensor_accessor_r_to_cpu_if_necessary(input, cpu_allocator);
  
  GenericTensorAccessorW output_cpu = cpu_allocator.allocate_tensor(output_shape);

  DataTypeDispatch1<CPUTensorAccessorBroadcast>{}(input.data_type, input_cpu, output_cpu);

  copy_accessor_data_to_l_from_r(output, output_cpu);
}

GenericTensorAccessorW
  tensor_accessor_broadcast(GenericTensorAccessorR const &input,
                            TensorDims const &output_dims,
                            Allocator &output_allocator) {

  TensorShape output_shape = TensorShape{output_dims, input.data_type};

  GenericTensorAccessorW output = output_allocator.allocate_tensor(output_shape);

  tensor_accessor_broadcast_to(input, output_dims, output);

  return output;
}

template <DataType DT>
struct CPUTensorAccessorTranspose {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    ASSERT(input.shape.num_dims() == 2);
    ASSERT(output.shape.num_dims() == 2);

    for (ArrayCoord const &input_coord : get_array_coord_set(input.shape)) {
      ASSERT(input_coord.ff_ordered.size() == 2);

      ArrayCoord output_coord = ArrayCoord{
        reversed(input_coord.ff_ordered),
      };

      output.at<DT>(output_coord.ff_ordered) = input.at<DT>(input_coord.ff_ordered);
    }
  }
};

static TensorShape get_transpose_output_shape(TensorShape const &input_shape) {
  return TensorShape{
    TensorDims{
      reversed(input_shape.dims.ff_ordered),
    },
    input_shape.data_type,
  };
}

void tensor_accessor_transpose_to(GenericTensorAccessorR const &input,
                                  GenericTensorAccessorW const &output) {
  ASSERT(input.shape.num_dims() == 2);

  TensorShape output_shape = get_transpose_output_shape(get_tensor_shape_for_accessor_r(input));

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR input_cpu = copy_tensor_accessor_r_to_cpu_if_necessary(input, cpu_allocator);
  
  GenericTensorAccessorW output_cpu = cpu_allocator.allocate_tensor(output_shape);

  DataTypeDispatch1<CPUTensorAccessorTranspose>{}(input.data_type, input_cpu, output_cpu);

  copy_accessor_data_to_l_from_r(output, output_cpu);
} 

GenericTensorAccessorW
  tensor_accessor_transpose(GenericTensorAccessorR const &input,
                            Allocator &output_allocator) {

  TensorShape output_shape = get_transpose_output_shape(get_tensor_shape_for_accessor_r(input));

  GenericTensorAccessorW output = output_allocator.allocate_tensor(output_shape);

  tensor_accessor_transpose_to(input, output);

  return output;
}

template <DataType DT>
struct CPUTensorAccessorReduce {
  void operator()(GenericTensorAccessorR const &input,
                  ff_dim_t reduction_dim,
                  GenericTensorAccessorW const &output) {
    fill_with_zeros(output);

    for (ArrayCoord const &input_coord : get_array_coord_set(input.shape)) {
      ArrayCoord output_coord = array_coord_drop_dims(input_coord, 
                                                      [&](ff_dim_t input_coord_dim) {
                                                        return input_coord_dim == reduction_dim;
                                                      });

      output.at<DT>(output_coord.ff_ordered) += input.at<DT>(input_coord.ff_ordered);
    }
  }
};

static TensorShape get_reduce_output_shape(TensorShape const &input_shape, ff_dim_t reduction_dim) {
  ASSERT(tensor_dims_has_dim(input_shape.dims, reduction_dim), 
         input_shape.dims, reduction_dim);

  return TensorShape{
    TensorDims{
      concat(
        slice(input_shape.dims.ff_ordered, ff_dim_t{0_n}, reduction_dim),
        slice(input_shape.dims.ff_ordered, ff_dim_t{reduction_dim.value + 1_n}, std::nullopt)),
    },
    input_shape.data_type,
  };
}

void tensor_accessor_reduce_to(GenericTensorAccessorR const &input,
                               ff_dim_t reduction_dim,
                               GenericTensorAccessorW const &output) {

  TensorShape output_shape = get_reduce_output_shape(get_tensor_shape_for_accessor_r(input), reduction_dim);

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR input_cpu = copy_tensor_accessor_r_to_cpu_if_necessary(input, cpu_allocator);
  
  GenericTensorAccessorW output_cpu = cpu_allocator.allocate_tensor(output_shape);

  DataTypeDispatch1<CPUTensorAccessorReduce>{}(input.data_type, input_cpu, reduction_dim, output_cpu);

  copy_accessor_data_to_l_from_r(output, output_cpu);
}

GenericTensorAccessorW
  tensor_accessor_reduce(GenericTensorAccessorR const &input,
                         ff_dim_t reduction_dim,
                         Allocator &output_allocator) {

  TensorShape output_shape = get_reduce_output_shape(get_tensor_shape_for_accessor_r(input), reduction_dim);

  GenericTensorAccessorW output = output_allocator.allocate_tensor(output_shape);

  tensor_accessor_reduce_to(input, reduction_dim, output);

  return output;
}

} // namespace FlexFlow
