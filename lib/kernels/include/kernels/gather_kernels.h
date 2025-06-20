#ifndef _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/gather_per_device_state.dtg.h"

namespace FlexFlow::Kernels::Gather {

std::optional<GatherPerDeviceState> init_kernel(DeviceType device_type,
                                                PerDeviceFFHandle const &handle,
                                                legion_dim_t legion_dim);

void forward_kernel(device_stream_t const &stream,
                    std::optional<GatherPerDeviceState> const &per_device_state,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorR const &index,
                    GenericTensorAccessorW const &output);

void backward_kernel(device_stream_t const &stream,
                     std::optional<GatherPerDeviceState> const &per_device_state,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &index,
                     GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow

#endif
