#ifndef _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/ff_handle.h"
#include "kernels/linear_per_device_state.dtg.h"
#include "op-attrs/datatype.h"
#include "op-attrs/ops/linear_attrs.dtg.h"
#include "pcg/device_type.dtg.h"

namespace FlexFlow {

std::optional<LinearPerDeviceState>
    linear_init_kernel(DeviceType device_type,
                       device_handle_t const &handle,
                       std::optional<Activation> activation,
                       std::optional<RegularizerAttrs> regularizer,
                       bool use_bias,
                       DataType input_type,
                       DataType weight_type,
                       DataType output_type,
                       int batch_size,
                       int output_num_channels);

void linear_forward_kernel(
    device_stream_t const &stream,
    std::optional<LinearPerDeviceState> const &per_device_state,
    LinearAttrs const &attrs,
    GenericTensorAccessorR const &input_accessor,
    GenericTensorAccessorW const &output_accessor,
    GenericTensorAccessorR const &projection_accessor,
    std::optional<GenericTensorAccessorR> const &bias_accessor);

void linear_backward_kernel(
    device_stream_t const &stream,
    std::optional<LinearPerDeviceState> const &per_device_state,
    LinearAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &projection,
    GenericTensorAccessorW const &projection_grad,
    std::optional<GenericTensorAccessorW> const &bias_grad);

void linear_cleanup_kernel(
    DeviceType device_type,
    std::optional<LinearPerDeviceState> &per_device_state);

} // namespace FlexFlow

#endif
