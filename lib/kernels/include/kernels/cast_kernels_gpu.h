#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CAST_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CAST_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"

namespace FlexFlow::Kernels::Cast {

void gpu_forward_kernel(ffStream_t stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void gpu_backward_kernel(ffStream_t stream,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow

#endif
