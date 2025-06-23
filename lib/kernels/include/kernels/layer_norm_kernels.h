#ifndef _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H

#include "kernels/allocation.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/ff_handle.h"
#include "kernels/layer_norm_per_device_state.dtg.h"

namespace FlexFlow::Kernels::LayerNorm {

std::optional<LayerNormPerDeviceState>
    init_kernel(DeviceType device_type,
                PerDeviceFFHandle const &handle,
                Allocator &allocator,
                bool elementwise_affine,
                int64_t effective_batch_size,
                int64_t effective_num_elements,
                float eps);

void forward_kernel(
    device_stream_t const &stream,
    std::optional<LayerNormPerDeviceState> const &per_device_state,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output,
    GenericTensorAccessorW const &gamma,
    GenericTensorAccessorW const &beta);

void backward_kernel(
    device_stream_t const &stream,
    std::optional<LayerNormPerDeviceState> const &per_device_state,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &gamma,
    GenericTensorAccessorW const &gamma_grad,
    GenericTensorAccessorW const &beta_grad);

void cleanup_kernel(
    DeviceType device_type,
    std::optional<LayerNormPerDeviceState> const &per_device_state);

} // namespace FlexFlow::Kernels::LayerNorm

#endif // _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H
