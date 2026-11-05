#ifndef _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/ff_handle.h"
#include "kernels/reduce_per_device_state.dtg.h"
#include "op-attrs/operator_type.dtg.h"
#include "op-attrs/ops/reduce_attrs.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "pcg/device_type.dtg.h"

namespace FlexFlow {

std::optional<ReducePerDeviceState>
    reduce_init_kernel(DeviceType device_type,
                       device_handle_t const &handle,
                       ReduceAttrs const &attrs,
                       TensorShape const &input_shape);

void reduce_forward_kernel(
    device_stream_t const &stream,
    std::optional<ReducePerDeviceState> const &per_device_state,
    ReduceAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output);

void reduce_backward_kernel(
    device_stream_t const &stream,
    std::optional<ReducePerDeviceState> const &per_device_state,
    ReduceAttrs const &attrs,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H
