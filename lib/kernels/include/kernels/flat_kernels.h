#ifndef _FLEXFLOW_OPS_KERNELS_FLAT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_FLAT_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"

namespace FlexFlow::Kernels::Flat {

void forward_kernel(device_stream_t const &stream,
                    GenericTensorAccessorR const &input,
                    float *output_ptr);

void backward_kernel(device_stream_t const &stream,
                     GenericTensorAccessorR const &input,
                     float const *output_grad_ptr,
                     float *input_grad_ptr);

} // namespace FlexFlow::Kernels::Flat

#endif // _FLEXFLOW_OPS_KERNELS_FLAT_KERNELS_H
