#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REDUCE_TENSOR_ACCESSORS_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REDUCE_TENSOR_ACCESSORS_H

#include "kernels/accessor.h"
#include "kenrels/allocation.h"

namespace FlexFlow {



template <typename DTIn, typename DTOut>
struct CPUReduceTensorAccessorInDims {
  template <typename F>
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW &output,
                  std::unordered_set<ff_dim_t> const &dims_to_reduce,
                  F &&f) {
    
    ASSERT(input.device_type == DeviceType::CPU);
    ASSERT(output.device_type == DeviceType::CPU);

    for (ArrayCoord const &coord : get_array_coord_set(input.shape)) {
      output.at<type_to_data_type_enum_v<DTOut>>(coord)
    }
  }
};

template <typename F>
GenericTensorAccessorW reduce_tensor_accessor_in_dims(std::unordered_set<ff_dim_t> const &dims,
                                                      F &&f) {
  
}

GenericTensorAccessorW reduce_tensor_accessor_all(GenericTensorAcessorR const &input,
                                                  Allocator &allocator);

} // namespace FlexFlow

#endif
