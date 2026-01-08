#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ELEMENT_BINARY_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ELEMENT_BINARY_KERNELS_GPU_H

#include "kernels/element_binary_per_device_state.dtg.h"
#include "op-attrs/operator_type.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow::Kernels::ElementBinary {

ElementBinaryPerDeviceState gpu_init_kernel(PerDeviceFFHandle handle,
                                            OperatorType op_type,
                                            bool should_broadcast_lhs,
                                            bool should_broadcast_rhs,
                                            TensorShape const &lhs_shape,
                                            TensorShape const &rhs_shape,
                                            TensorShape const &output_shape);

void gpu_forward_kernel(ffStream_t stream,
                        ElementBinaryPerDeviceState const &per_device_state,
                        float const *lhs_ptr,
                        float const *rhs_ptr,
                        float *out_ptr,
                        OperatorType op_type,
                        bool broadcast_inputLHS,
                        PerDeviceFFHandle handle);

void gpu_backward_kernel(ffStream_t stream,
                         ElementBinaryPerDeviceState const &per_device_state,
                         float const *out_grad_ptr,
                         float const *lhs_ptr,
                         float const *rhs_ptr,
                         float *lhs_grad_ptr,
                         float *rhs_grad_ptr,
                         OperatorType op_type,
                         bool broadcast_inputLHS,
                         bool broadcast_inputRHS,
                         PerDeviceFFHandle handle);

void gpu_cleanup_kernel(ElementBinaryPerDeviceState const &per_device_state);

} // namespace FlexFlow::Kernels::ElementBinary

#endif
