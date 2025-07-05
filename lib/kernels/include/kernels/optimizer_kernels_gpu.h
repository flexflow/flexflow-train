#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_OPTIMIZER_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_OPTIMIZER_KERNELS_GPU_H

#include "kernels/device.h"
#include "kernels/ff_handle.h"

namespace FlexFlow {

void gpu_sgd_ps_update_task(ffStream_t stream,
                            float lr,
                            float momentum,
                            bool nesterov,
                            float weight_decay,
                            float const *weight_grad_ptr,
                            size_t size,
                            int num_replicas,
                            float *weight_ptr,
                            float *sgd_v_ptr);

void gpu_sgd_nccl_update_task(ffStream_t stream,
                              float lr,
                              float momentum,
                              bool nesterov,
                              float weight_decay,
                              PerDeviceFFHandle const &,
                              float const *weight_grad_ptr,
                              size_t size,
                              float *weight_ptr,
                              float *sgd_v_ptr);

void gpu_adam_ps_update_task(ffStream_t stream,
                             float alpha_t,
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

void gpu_adam_nccl_update_task(ffStream_t stream,
                               float alpha_t,
                               float beta1,
                               float beta2,
                               float weight_decay,
                               float epsilon,
                               PerDeviceFFHandle const &handle,
                               float const *weight_grad_ptr,
                               size_t size,
                               float *weight_ptr,
                               float *adam_v_ptr,
                               float *adam_m_ptr);

} // namespace FlexFlow

#endif
