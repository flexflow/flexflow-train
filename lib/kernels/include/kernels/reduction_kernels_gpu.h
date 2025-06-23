#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REDUCTION_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REDUCTION_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"

namespace FlexFlow::Kernels::Reduction {

void gpu_forward_kernel(ffStream_t stream,
                        GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output,
                        size_t num_replicas);

void gpu_backward_kernel(ffStream_t stream,
                         GenericTensorAccessorR const &output,
                         GenericTensorAccessorW const &input);

} // namespace FlexFlow::Kernels::Reduction

#endif
