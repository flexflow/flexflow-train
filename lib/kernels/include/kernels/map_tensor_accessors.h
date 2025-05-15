#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_MAP_TENSOR_ACCESSORS_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_MAP_TENSOR_ACCESSORS_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/datatype_dispatch.h"
#include "utils/containers/require_same.h"
#include "utils/containers/require_all_same1.h"

namespace FlexFlow {

template <DataType DT>
struct CPUMapTensorAccessor {
  template <typename F>
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW &output,
                  F &&f) {
    ArrayShape shape = require_same(input.shape, output.shape);

    ASSERT(input.device_type == DeviceType::CPU);
    ASSERT(output.device_type == DeviceType::CPU);

    for (ArrayCoord const &coord : get_array_coord_set(shape)) {
      output.at<DataType::BOOL>(coord.ff_ordered) 
        = f(input.at<DT>(coord.ff_ordered));
    }
  }
};

template <typename F, typename Out = std::invoke_result_t<F, float>>
GenericTensorAccessorW map_tensor_accessor(GenericTensorAccessorR const &input,
                                           Allocator &output_allocator,
                                           F &&f) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR input_cpu = copy_tensor_accessor_r_to_cpu_if_necessary(input, cpu_allocator);

  GenericTensorAccessorW output_cpu = cpu_allocator.allocate_tensor(get_tensor_shape(input.shape, type_to_data_type_enum_v<Out>));

  DataTypeDispatch1<CPUMapTensorAccessor>{}(input.data_type, input_cpu, output_cpu, f);

  return copy_tensor_accessor_w(output_cpu, output_allocator);
}

template <DataType DT>
struct CPUMapTensorAccessors2 {
  template <typename F, typename Out = std::invoke_result_t<F, float, float>>
  void operator()(GenericTensorAccessorR const &lhs,
                  GenericTensorAccessorR const &rhs,
                  GenericTensorAccessorW &output,
                  F &&f) {

    ArrayShape shape = throw_if_unexpected(require_all_same1(std::vector{
      lhs.shape,
      rhs.shape,
      output.shape,
    }));

    ASSERT(lhs.device_type == DeviceType::CPU);
    ASSERT(rhs.device_type == DeviceType::CPU);
    ASSERT(output.device_type == DeviceType::CPU);

    for (ArrayCoord const &coord : get_array_coord_set(shape)) {
      output.at<type_to_data_type_enum_v<Out>>(coord.ff_ordered) 
        = f(lhs.at<DT>(coord.ff_ordered), rhs.at<DT>(coord.ff_ordered));
    }
  }
};

template <typename F, typename Out = std::invoke_result_t<F, float, float>>
GenericTensorAccessorW map_tensor_accessors2(GenericTensorAccessorR const &lhs,
                                             GenericTensorAccessorR const &rhs,
                                             Allocator &output_allocator,
                                             F &&f) {
  ArrayShape shape = require_same(lhs.shape, rhs.shape);
  DataType input_data_type = require_same(lhs.data_type, rhs.data_type);

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR lhs_cpu = copy_tensor_accessor_r_to_cpu_if_necessary(lhs, cpu_allocator);
  GenericTensorAccessorR rhs_cpu = copy_tensor_accessor_r_to_cpu_if_necessary(rhs, cpu_allocator);
  DataType output_data_type = type_to_data_type_enum_v<Out>;
  GenericTensorAccessorW output_cpu = cpu_allocator.allocate_tensor(get_tensor_shape(shape, output_data_type));

  DataTypeDispatch1<CPUMapTensorAccessors2>{}(input_data_type, lhs_cpu, rhs_cpu, output_cpu, f);

  return copy_tensor_accessor_w(output_cpu, output_allocator);
}


} // namespace FlexFlow

#endif
