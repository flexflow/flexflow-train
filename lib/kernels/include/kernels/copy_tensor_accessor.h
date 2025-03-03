#ifndef _FLEXFLOW_KERNELS_COPY_TENSOR_ACCESSOR_H
#define _FLEXFLOW_KERNELS_COPY_TENSOR_ACCESSOR_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"

namespace FlexFlow {

void copy_accessor_data_to_l_from_r(GenericTensorAccessorW &dst_accessor,
                                    GenericTensorAccessorR const &src_accessor);

GenericTensorAccessorR
    copy_tensor_accessor_r(GenericTensorAccessorR const &src_accessor,
                           Allocator &allocator);

GenericTensorAccessorW
    copy_tensor_accessor_w(GenericTensorAccessorW const &src_accessor,
                           Allocator &allocator);

GenericTensorAccessorW
    copy_accessor_w_to_cpu_if_necessary(GenericTensorAccessorW const &accessor,
                                        Allocator &allocator);

GenericTensorAccessorR
    copy_accessor_r_to_cpu_if_necessary(GenericTensorAccessorR const &accessor,
                                        Allocator &allocator);

} // namespace FlexFlow

#endif
