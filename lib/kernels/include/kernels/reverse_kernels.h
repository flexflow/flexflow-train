#ifndef _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"
#include "op-attrs/ops/reverse_attrs.dtg.h"

namespace FlexFlow::Kernels::Reverse {

void forward_kernel(device_stream_t const &stream,
                    GenericTensorAccessorR const &input_accessor,
                    GenericTensorAccessorW &output_accessor,
                    ReverseAttrs const &);

void backward_kernel(device_stream_t const &stream,
                    GenericTensorAccessorR const &output_accessor,
                    GenericTensorAccessorW &input_accessor,
                    ReverseAttrs const &);

} // namespace FlexFlow::Kernels::Reverse

#endif // _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_H
