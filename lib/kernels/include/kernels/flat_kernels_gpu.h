#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_FLAT_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_FLAT_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"

namespace FlexFlow::Kernels::Flat {

void gpu_forward_kernel(ffStream_t stream,
                        GenericTensorAccessorR const &input,
                        float *output_ptr);

void gpu_backward_kernel(ffStream_t stream,
                         GenericTensorAccessorR const &input,
                         float const *output_grad_ptr,
                         float *input_grad_ptr);

} // namespace FlexFlow::Kernels::Flat

#endif
