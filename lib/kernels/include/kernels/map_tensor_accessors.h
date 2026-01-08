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

    for (TensorDimsCoord const &coord :
         get_tensor_dims_coord_set(accessor.shape.dims)) {
      accessor.at<DT>(coord) = f(accessor.at<DT>(coord));
    }
  }
};

template <typename F>
void map_tensor_accessor_inplace(GenericTensorAccessorW const &accessor,
                                 F &&f) {
  ASSERT(accessor.device_type == DeviceType::CPU);

  DataTypeDispatch1<CPUMapTensorAccessorInPlace>{}(
      accessor.shape.data_type, accessor, f);
}

template <DataType DT>
struct CPUMapTensorAccessor {
  template <typename F>
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output,
                  F &&f) {
    TensorDims tensor_dims = require_same(input.shape.dims, output.shape.dims);

    ASSERT(input.device_type == DeviceType::CPU);
    ASSERT(output.device_type == DeviceType::CPU);

    for (TensorDimsCoord const &coord :
         get_tensor_dims_coord_set(tensor_dims)) {
      output.at<
          type_to_data_type_enum_v<std::invoke_result_t<F, real_type_t<DT>>>>(
          coord) = f(input.at<DT>(coord));
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
      cpu_allocator.allocate_tensor(output.shape);

  DataTypeDispatch1<CPUMapTensorAccessor>{}(
      input.shape.data_type, input_cpu, output_cpu, f);

  copy_accessor_data_to_l_from_r(
      output, read_only_accessor_from_write_accessor(output_cpu));
}

template <typename F, typename Out = std::invoke_result_t<F, float>>
GenericTensorAccessorW map_tensor_accessor(GenericTensorAccessorR const &input,
                                           F &&f,
                                           Allocator &output_allocator) {
  TensorShape output_shape = TensorShape{
      /*dims=*/input.shape.dims,
      /*data_type=*/type_to_data_type_enum_v<Out>,
  };

  GenericTensorAccessorW output =
      output_allocator.allocate_tensor(output_shape);

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

    TensorDims dims = require_all_same1(std::vector{
        lhs.shape.dims,
        rhs.shape.dims,
        output.shape.dims,
    });

    ASSERT(lhs.device_type == DeviceType::CPU);
    ASSERT(rhs.device_type == DeviceType::CPU);
    ASSERT(output.device_type == DeviceType::CPU);

    for (TensorDimsCoord const &coord : get_tensor_dims_coord_set(dims)) {
      output.at<type_to_data_type_enum_v<Out>>(coord) =
          f(lhs.at<DTL>(coord), rhs.at<DTR>(coord));
    }
  }
};

template <typename F>
void map_tensor_accessors2_to(GenericTensorAccessorR const &lhs,
                              GenericTensorAccessorR const &rhs,
                              DataType output_data_type,
                              F &&f,
                              GenericTensorAccessorW const &output) {
  TensorDims output_dims = require_same(lhs.shape.dims, rhs.shape.dims);

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR lhs_cpu =
      copy_tensor_accessor_r_to_cpu_if_necessary(lhs, cpu_allocator);
  GenericTensorAccessorR rhs_cpu =
      copy_tensor_accessor_r_to_cpu_if_necessary(rhs, cpu_allocator);
  GenericTensorAccessorW output_cpu =
      cpu_allocator.allocate_tensor(TensorShape{output_dims, output_data_type});

  DataTypeDispatch2<CPUMapTensorAccessors2>{}(lhs.shape.data_type,
                                              rhs.shape.data_type,
                                              lhs_cpu,
                                              rhs_cpu,
                                              output_cpu,
                                              f);

  return copy_accessor_data_to_l_from_r(output, output_cpu);
}

template <typename F>
GenericTensorAccessorW map_tensor_accessors2(GenericTensorAccessorR const &lhs,
                                             GenericTensorAccessorR const &rhs,
                                             DataType output_data_type,
                                             F &&f,
                                             Allocator &output_allocator) {
  TensorDims output_dims = require_same(lhs.shape.dims, rhs.shape.dims);

  GenericTensorAccessorW output = output_allocator.allocate_tensor(
      TensorShape{output_dims, output_data_type});

  map_tensor_accessors2_to(lhs, rhs, output_data_type, f, output);

  return output;
}

} // namespace FlexFlow

#endif
