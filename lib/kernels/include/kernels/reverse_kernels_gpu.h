#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REVERSE_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REVERSE_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "op-attrs/ops/reverse_attrs.dtg.h"

namespace FlexFlow::Kernels::Reverse {

void gpu_forward_kernel(ffStream_t stream,
                        GenericTensorAccessorR const &input_accessor,
                        GenericTensorAccessorW &output_accessor,
                        ReverseAttrs const &);

void gpu_backward_kernel(ffStream_t stream,
                         GenericTensorAccessorR const &output_accessor,
                         GenericTensorAccessorW &input_accessor,
                         ReverseAttrs const &);

} // namespace FlexFlow::Kernels::Reverse

#endif
