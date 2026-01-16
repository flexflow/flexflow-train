#include "kernels/element_binary_kernels.h"
#include "kernels/element_binary_kernels_cpu.h"
#include "kernels/element_binary_kernels_gpu.h"
#include <libassert/assert.hpp>

namespace FlexFlow::Kernels::ElementBinary {

std::optional<ElementBinaryPerDeviceState>
    init_kernel(DeviceType device_type,
                device_handle_t const &handle,
                OperatorType op_type,
                bool should_broadcast_lhs,
                bool should_broadcast_rhs,
                TensorShape const &lhs_shape,
                TensorShape const &rhs_shape,
                TensorShape const &output_shape) {
  if (device_type == DeviceType::GPU) {
    return gpu_init_kernel(
        /*handle=*/handle.require_for_gpu(),
        /*op_type=*/op_type,
        /*should_broadcast_lhs=*/should_broadcast_lhs,
        /*should_broadcast_rhs=*/should_broadcast_rhs,
        /*lhs_shape=*/lhs_shape,
        /*rhs_shape=*/rhs_shape,
        /*output_shape=*/output_shape);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(handle.is_for_cpu());
    return std::nullopt;
  }
}

void forward_kernel(
    device_stream_t const &stream,
    std::optional<ElementBinaryPerDeviceState> const &per_device_state,
    float const *lhs_ptr,
    float const *rhs_ptr,
    float *out_ptr,
    OperatorType op_type,
    bool broadcast_inputLHS,
    device_handle_t const &handle) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*lhs_ptr=*/lhs_ptr,
        /*rhs_ptr=*/rhs_ptr,
        /*out_ptr=*/out_ptr,
        /*op_type=*/op_type,
        /*broadcast_inputLHS=*/broadcast_inputLHS,
        /*handle=*/handle.require_for_gpu());
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    ASSERT(handle.is_for_cpu());
    cpu_forward_kernel(
        /*lhs_ptr=*/lhs_ptr,
        /*rhs_ptr=*/rhs_ptr,
        /*out_ptr=*/out_ptr,
        /*op_type=*/op_type,
        /*broadcast_inputLHS=*/broadcast_inputLHS);
  }
}

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
    device_handle_t const &handle) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*out_grad_ptr=*/out_grad_ptr,
        /*lhs_ptr=*/lhs_ptr,
        /*rhs_ptr=*/rhs_ptr,
        /*lhs_grad_ptr=*/lhs_grad_ptr,
        /*rhs_grad_ptr=*/rhs_grad_ptr,
        /*op_type=*/op_type,
        /*broadcast_inputLHS=*/broadcast_inputLHS,
        /*broadcast_inputRHS=*/broadcast_inputRHS,
        /*handle=*/handle.require_for_gpu());
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    ASSERT(handle.is_for_cpu());
    cpu_backward_kernel(
        /*out_grad_ptr=*/out_grad_ptr,
        /*lhs_ptr=*/lhs_ptr,
        /*rhs_ptr=*/rhs_ptr,
        /*lhs_grad_ptr=*/lhs_grad_ptr,
        /*rhs_grad_ptr=*/rhs_grad_ptr,
        /*op_type=*/op_type,
        /*broadcast_inputLHS=*/broadcast_inputLHS,
        /*broadcast_inputRHS=*/broadcast_inputRHS);
  }
}

void cleanup_kernel(
    DeviceType device_type,
    std::optional<ElementBinaryPerDeviceState> const &per_device_state) {
  if (device_type == DeviceType::GPU) {
    gpu_cleanup_kernel(per_device_state.value());
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(per_device_state == std::nullopt);
  }
}

} // namespace FlexFlow::Kernels::ElementBinary
