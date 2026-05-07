#include "kernels/element_binary_kernels_cpu.h"
#include "op-attrs/operator_type.dtg.h"
#include "utils/exception.h"

namespace FlexFlow::Kernels::ElementBinary {

void cpu_forward_kernel(float const *lhs_ptr,
                        float const *rhs_ptr,
                        float *out_ptr,
                        OperatorType op_type,
                        bool broadcast_inputLHS,
                        size_t num_elements) {
  switch (op_type) {
    case OperatorType::EW_ADD:
      for (size_t i = 0; i < num_elements; i++) {
        out_ptr[i] = lhs_ptr[i] + rhs_ptr[i];
      }
      break;
    case OperatorType::EW_SUB:
      for (size_t i = 0; i < num_elements; i++) {
        out_ptr[i] = lhs_ptr[i] - rhs_ptr[i];
      }
      break;
    case OperatorType::EW_MUL:
      for (size_t i = 0; i < num_elements; i++) {
        out_ptr[i] = lhs_ptr[i] * rhs_ptr[i];
      }
      break;
    case OperatorType::EW_DIV:
      for (size_t i = 0; i < num_elements; i++) {
        out_ptr[i] = lhs_ptr[i] / rhs_ptr[i];
      }
      break;
    default:
      NOT_IMPLEMENTED();
  }
}

void cpu_backward_kernel(float const *out_grad_ptr,
                         float const *lhs_ptr,
                         float const *rhs_ptr,
                         float *lhs_grad_ptr,
                         float *rhs_grad_ptr,
                         OperatorType op_type,
                         bool broadcast_inputLHS,
                         bool broadcast_inputRHS,
                         size_t num_elements) {
  switch (op_type) {
    case OperatorType::EW_ADD:
    case OperatorType::EW_SUB:
      for (size_t i = 0; i < num_elements; i++) {
        lhs_grad_ptr[i] += out_grad_ptr[i];
        rhs_grad_ptr[i] += (op_type == OperatorType::EW_SUB) ? -out_grad_ptr[i]
                                                             : out_grad_ptr[i];
      }
      break;
    case OperatorType::EW_MUL:
      for (size_t i = 0; i < num_elements; i++) {
        lhs_grad_ptr[i] += out_grad_ptr[i] * rhs_ptr[i];
        rhs_grad_ptr[i] += out_grad_ptr[i] * lhs_ptr[i];
      }
      break;
    default:
      NOT_IMPLEMENTED();
  }
}
} // namespace FlexFlow::Kernels::ElementBinary
