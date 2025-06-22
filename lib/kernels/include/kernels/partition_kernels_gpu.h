#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_PARTITION_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_PARTITION_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "kernels/partition_per_device_state.dtg.h"

namespace FlexFlow::Kernels::Repartition {

RepartitionPerDeviceState gpu_init_kernel(PerDeviceFFHandle const &handle,
                                      DataType data_type);

void gpu_forward_kernel(ffStream_t stream,
                    RepartitionPerDeviceState const &per_device_state,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void gpu_backward_kernel(ffStream_t stream,
                     RepartitionPerDeviceState const &per_device_state,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const &input_grad);

void gpu_cleanup_kernel(RepartitionPerDeviceState &per_device_state);

} // namespace FlexFlow

#endif
