#ifndef _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_H

#include "kernels/device.h"
#include "kernels/reverse_kernels_cpu.h"

namespace FlexFlow::Kernels::Reverse {

void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorR const &input_accessor,
                    GenericTensorAccessorW &output_accessor,
                    ReverseAttrs const &);

void backward_kernel(ffStream_t stream,
                     GenericTensorAccessorR const &output_accessor,
                     GenericTensorAccessorW &input_accessor,
                     ReverseAttrs const &);

} // namespace FlexFlow::Kernels::Reverse

#endif // _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_H
