#include "kernels/optimizer_kernels_cpu.h"
#include "utils/exception.h"

namespace FlexFlow {

void cpu_sgd_update_task(float lr,
                         float momentum,
                         bool nesterov,
                         float weight_decay,
                         float const *weight_grad_ptr,
                         size_t size,
                         int num_replicas,
                         float *weight_ptr,
                         float *sgd_v_ptr) {
  NOT_IMPLEMENTED();
}

void cpu_adam_update_task(float alpha_t,
                          float beta1,
                          float beta2,
                          float weight_decay,
                          float epsilon,
                          float const *weight_grad_ptr,
                          size_t size,
                          int num_replicas,
                          float *weight_ptr,
                          float *adam_v_ptr,
                          float *adam_m_ptr) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
