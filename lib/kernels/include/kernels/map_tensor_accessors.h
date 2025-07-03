#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_MAP_TENSOR_ACCESSORS_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_MAP_TENSOR_ACCESSORS_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/local_cpu_allocator.h"
#include "utils/containers/require_all_same1.h"
#include "utils/containers/require_same.h"
#include <type_traits>

namespace FlexFlow {

template <DataType DT>
struct CPUMapTensorAccessorInPlace {
  template <typename F>
  void operator()(GenericTensorAccessorW const &accessor, F &&f) {
    ASSERT(accessor.device_type == DeviceType::CPU);

    for (ArrayCoord const &coord : get_array_coord_set(accessor.shape)) {
      accessor.at<DT>(coord.ff_ordered) = f(accessor.at<DT>(coord.ff_ordered));
    }
  }
};

template <typename F>
void map_tensor_accessor_inplace(GenericTensorAccessorW const &accessor,
                                 F &&f) {
  ASSERT(accessor.device_type == DeviceType::CPU);

  DataTypeDispatch1<CPUMapTensorAccessorInPlace>{}(
      accessor.data_type, accessor, f);
}

template <DataType DT>
struct CPUMapTensorAccessor {
  template <typename F>
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output,
                  F &&f) {
    ArrayShape shape = require_same(input.shape, output.shape);

    ASSERT(input.device_type == DeviceType::CPU);
    ASSERT(output.device_type == DeviceType::CPU);

    for (ArrayCoord const &coord : get_array_coord_set(shape)) {
      output.at<
          type_to_data_type_enum_v<std::invoke_result_t<F, real_type_t<DT>>>>(
          coord.ff_ordered) = f(input.at<DT>(coord.ff_ordered));
    }
  }
};

template <typename F, typename Out = std::invoke_result_t<F, float>>
void map_tensor_accessor_to(GenericTensorAccessorR const &input,
                            F &&f,
                            GenericTensorAccessorW const &output) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR input_cpu =
      copy_tensor_accessor_r_to_cpu_if_necessary(input, cpu_allocator);

  GenericTensorAccessorW output_cpu =
      cpu_allocator.allocate_tensor(tensor_shape_from_array_shape(
          input.shape, type_to_data_type_enum_v<Out>));

  DataTypeDispatch1<CPUMapTensorAccessor>{}(
      input.data_type, input_cpu, output_cpu, f);

  copy_accessor_data_to_l_from_r(
      output, read_only_accessor_from_write_accessor(output_cpu));
}

template <typename F, typename Out = std::invoke_result_t<F, float>>
GenericTensorAccessorW map_tensor_accessor(GenericTensorAccessorR const &input,
                                           F &&f,
                                           Allocator &output_allocator) {
  GenericTensorAccessorW output =
      output_allocator.allocate_tensor(tensor_shape_from_array_shape(
          input.shape, type_to_data_type_enum_v<Out>));

  map_tensor_accessor_to(input, f, output);

  return output;
}

template <DataType DTL, DataType DTR>
struct CPUMapTensorAccessors2 {
  template <typename F,
            typename Out =
                std::invoke_result_t<F, real_type_t<DTL>, real_type_t<DTR>>>
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
      output.at<type_to_data_type_enum_v<Out>>(coord.ff_ordered) =
          f(lhs.at<DTL>(coord.ff_ordered), rhs.at<DTR>(coord.ff_ordered));
    }
  }
};

template <typename F>
void map_tensor_accessors2_to(GenericTensorAccessorR const &lhs,
                              GenericTensorAccessorR const &rhs,
                              DataType output_data_type,
                              F &&f,
                              GenericTensorAccessorW const &output) {
  ArrayShape shape = require_same(lhs.shape, rhs.shape);

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR lhs_cpu =
      copy_tensor_accessor_r_to_cpu_if_necessary(lhs, cpu_allocator);
  GenericTensorAccessorR rhs_cpu =
      copy_tensor_accessor_r_to_cpu_if_necessary(rhs, cpu_allocator);
  GenericTensorAccessorW output_cpu = cpu_allocator.allocate_tensor(
      tensor_shape_from_array_shape(shape, output_data_type));

  DataTypeDispatch2<CPUMapTensorAccessors2>{}(
      lhs.data_type, rhs.data_type, lhs_cpu, rhs_cpu, output_cpu, f);

  return copy_accessor_data_to_l_from_r(output, output_cpu);
}

template <typename F>
GenericTensorAccessorW map_tensor_accessors2(GenericTensorAccessorR const &lhs,
                                             GenericTensorAccessorR const &rhs,
                                             DataType output_data_type,
                                             F &&f,
                                             Allocator &output_allocator) {
  ArrayShape shape = require_same(lhs.shape, rhs.shape);

  GenericTensorAccessorW output = output_allocator.allocate_tensor(
      tensor_shape_from_array_shape(shape, output_data_type));

  map_tensor_accessors2_to(lhs, rhs, output_data_type, f, output);

  return output;
}

} // namespace FlexFlow

#endif
