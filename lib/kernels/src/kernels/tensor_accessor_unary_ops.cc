#include "kernels/tensor_accessor_unary_ops.h"
#include "kernels/array_coord.h"
#include "kernels/map_tensor_accessors.h"
#include "op-attrs/datatype_value.h"
#include "kernels/datatype_dispatch.h"
#include "op-attrs/ff_ordered/slice.h"
#include "op-attrs/tensor_dims.h"

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
        input_dims,
        output_dims,
        output_dims_coord);

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

  return copy_accessor_data_to_l_from_r(output_cpu, output);
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


} // namespace FlexFlow
