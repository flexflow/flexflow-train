#ifndef _FLEXFLOW_OPS_KERNELS_ELEMENT_UNARY_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_ELEMENT_UNARY_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/element_unary_per_device_state.dtg.h"
#include "kernels/ff_handle.h"
#include "op-attrs/ops/element_unary_attrs.dtg.h"

namespace FlexFlow::Kernels::ElementUnary {

std::optional<ElementUnaryPerDeviceState>
    init_kernel(DeviceType device_type,
                TensorShape const &input_shape,
                TensorShape const &output_shape,
                ElementUnaryAttrs const &attrs);

void forward_kernel(
    device_stream_t const &stream,
    std::optional<ElementUnaryPerDeviceState> const &device_state,
    ElementUnaryAttrs const &attrs,
    device_handle_t const &handle,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output);

void backward_kernel(
    device_stream_t const &stream,
    std::optional<ElementUnaryPerDeviceState> const &device_state,
    ElementUnaryAttrs const &attrs,
    device_handle_t const &handle,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad);

void cleanup_kernel(
    DeviceType device_type,
    std::optional<ElementUnaryPerDeviceState> &per_device_state);

} // namespace FlexFlow::Kernels::ElementUnary

#endif
