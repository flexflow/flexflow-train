#ifndef _FLEXFLOW_OPS_KERNELS_ELEMENT_BINARY_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_ELEMENT_BINARY_KERNELS_H

#include "kernels/array_shape.h"
#include "kernels/device.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/element_binary_per_device_state.dtg.h"
#include "kernels/ff_handle.h"
#include "op-attrs/datatype.h"
#include "op-attrs/operator_type.h"
#include "pcg/device_type.dtg.h"
#include "kernels/device_handle_t.dtg.h"

namespace FlexFlow::Kernels::ElementBinary {

std::optional<ElementBinaryPerDeviceState>
    init_kernel(DeviceType device_type,
                device_handle_t const &handle,
                OperatorType op_type,
                bool should_broadcast_lhs,
                bool should_broadcast_rhs,
                ArrayShape lhs_shape,
                ArrayShape rhs_shape,
                ArrayShape output_shape);

void forward_kernel(
    device_stream_t const &stream,
    std::optional<ElementBinaryPerDeviceState> const &per_device_state,
    float const *lhs_ptr,
    float const *rhs_ptr,
    float *out_ptr,
    OperatorType op_type,
    bool broadcast_inputLHS,
    device_handle_t const &handle);

void backward_kernel(
    device_stream_t const &stream,
    std::optional<ElementBinaryPerDeviceState> const &per_device_state,
    float const *out_grad_ptr,
    float const *lhs_ptr,
    float const *rhs_ptr,
    float *lhs_grad_ptr,
    float *rhs_grad_ptr,
    OperatorType op_type,
    bool broadcast_inputLHS,
    bool broadcast_inputRHS,
    device_handle_t const &handle);

void cleanup_kernel(
    DeviceType device_type,
    std::optional<ElementBinaryPerDeviceState> const &per_device_state);

} // namespace FlexFlow::Kernels::ElementBinary

#endif
