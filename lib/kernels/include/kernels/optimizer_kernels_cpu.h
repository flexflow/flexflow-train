#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_OPTIMIZER_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_OPTIMIZER_KERNELS_CPU_H

#include <cstddef>
#include "kernels/accessor.h"

namespace FlexFlow {

void cpu_sgd_update_task(float lr,
                         float momentum,
                         bool nesterov,
                         float weight_decay,
                         GenericTensorAccessorR const &weight_grad,
                         GenericTensorAccessorW const &weight,
                         std::optional<GenericTensorAccessorW> const &sgd_v);

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
                          float *adam_m_ptr);

} // namespace FlexFlow

#endif
