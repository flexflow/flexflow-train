#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ELEMENT_BINARY_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ELEMENT_BINARY_KERNELS_CPU_H

#include "op-attrs/operator_type.dtg.h"

namespace FlexFlow::Kernels::ElementBinary {

void cpu_forward_kernel(float const *lhs_ptr,
                    float const *rhs_ptr,
                    float *out_ptr,
                    OperatorType op_type,
                    bool broadcast_inputLHS);

void cpu_backward_kernel(float const *out_grad_ptr,
                     float const *lhs_ptr,
                     float const *rhs_ptr,
                     float *lhs_grad_ptr,
                     float *rhs_grad_ptr,
                     OperatorType op_type,
                     bool broadcast_inputLHS,
                     bool broadcast_inputRHS);

} // namespace FlexFlow

#endif
