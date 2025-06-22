#ifndef _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/partition_per_device_state.dtg.h"

namespace FlexFlow::Kernels::Repartition {

std::optional<RepartitionPerDeviceState> init_kernel(DeviceType device_type,
                                      PerDeviceFFHandle const &handle,
                                      DataType data_type);

void forward_kernel(device_stream_t const &stream,
                    std::optional<RepartitionPerDeviceState> const &per_device_state,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(device_stream_t const &stream,
                     std::optional<RepartitionPerDeviceState> const &per_device_state,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const &input_grad);

void cleanup_kernel(DeviceType device_type,
                    std::optional<RepartitionPerDeviceState> &per_device_state);

} // namespace Kernels::Repartition

#endif // _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H
