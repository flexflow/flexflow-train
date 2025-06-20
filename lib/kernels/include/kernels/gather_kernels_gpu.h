#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_GATHER_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_GATHER_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "kernels/gather_per_device_state.dtg.h"

namespace FlexFlow::Kernels::Gather {

GatherPerDeviceState gpu_init_kernel(PerDeviceFFHandle const &handle,
                                     legion_dim_t legion_dim);

void gpu_forward_kernel(ffStream_t stream,
                    GatherPerDeviceState const &per_device_state,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorR const &index,
                    GenericTensorAccessorW const &output);

void gpu_backward_kernel(ffStream_t stream,
                     GatherPerDeviceState const &per_device_state,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &index,
                     GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow

#endif
