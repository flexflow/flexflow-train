#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REPLICATE_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REPLICATE_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"

namespace FlexFlow::Kernels::Replicate {

void gpu_forward_kernel(ffStream_t stream,
                        GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output);

void gpu_backward_kernel(ffStream_t stream,
                         GenericTensorAccessorR const &output,
                         GenericTensorAccessorW const &input,
                         size_t num_replicas);

} // namespace FlexFlow::Kernels::Replicate

#endif
