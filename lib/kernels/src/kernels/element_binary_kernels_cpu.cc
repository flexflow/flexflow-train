#include "kernels/element_binary_kernels_cpu.h"
#include "utils/exception.h"

namespace FlexFlow::Kernels::ElementBinary {

void cpu_forward_kernel(float const *lhs_ptr,
                    float const *rhs_ptr,
                    float *out_ptr,
                    OperatorType op_type,
                    bool broadcast_inputLHS) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(float const *out_grad_ptr,
                     float const *lhs_ptr,
                     float const *rhs_ptr,
                     float *lhs_grad_ptr,
                     float *rhs_grad_ptr,
                     OperatorType op_type,
                     bool broadcast_inputLHS,
                     bool broadcast_inputRHS) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
