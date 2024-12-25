#ifndef _FLEXFLOW_KERNELS_COPY_TENSOR_ACCESSOR_H
#define _FLEXFLOW_KERNELS_COPY_TENSOR_ACCESSOR_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"

namespace FlexFlow {

GenericTensorAccessorR
    copy_tensor_accessor_r(GenericTensorAccessorR const &src_accessor,
                           Allocator &allocator);

GenericTensorAccessorW
    copy_tensor_accessor_w(GenericTensorAccessorW const &src_accessor,
                           Allocator &allocator);

} // namespace FlexFlow

#endif
