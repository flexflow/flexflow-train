#include "kernels/optimizer_kernels.h"
#include "kernels/optimizer_kernels_cpu.h"
#include "kernels/optimizer_kernels_gpu.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

void sgd_update_task(device_stream_t const &stream,
                        PerDeviceFFHandle const &handle,
                        float lr,
                        float momentum,
                        bool nesterov,
                        float weight_decay,
                        float const *weight_grad_ptr,
                        size_t size,
                        int num_replicas,
                        float *weight_ptr,
                        float *sgd_v_ptr) {
  if (stream.is_gpu()) {
    gpu_sgd_nccl_update_task(
                           /*stream=*/stream.require_gpu(),
                           /*lr=*/lr,
                           /*momentum=*/momentum,
                           /*nesterov=*/nesterov,
                           /*weight_decay=*/weight_decay,
                           /*handle=*/handle,
                           /*weight_grad_ptr=*/weight_grad_ptr,
                           /*size=*/size,
                           /*weight_ptr=*/weight_ptr,
                           /*sgd_v_ptr=*/sgd_v_ptr);
  } else {
    ASSERT(stream.is_cpu());
    cpu_sgd_update_task(
                           /*lr=*/lr,
                           /*momentum=*/momentum,
                           /*nesterov=*/nesterov,
                           /*weight_decay=*/weight_decay,
                           /*weight_grad_ptr=*/weight_grad_ptr,
                           /*size=*/size,
                           /*num_replicas=*/num_replicas,
                           /*weight_ptr=*/weight_ptr,
                           /*sgd_v_ptr=*/sgd_v_ptr);
  }
}

void adam_update_task(device_stream_t const &stream,
                         PerDeviceFFHandle const &handle,
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
                         float *adam_m_ptr) {
  if (stream.is_gpu()) {
    ASSERT(stream.is_cpu());
    gpu_adam_nccl_update_task(
                              /*stream=*/stream.require_gpu(),
                              /*alpha_t=*/alpha_t,
                              /*beta1=*/beta1,
                              /*beta2=*/beta2,
                              /*weight_decay=*/weight_decay,
                              /*epsilon=*/epsilon,
                              /*handle=*/handle,
                              /*weight_grad_ptr=*/weight_grad_ptr,
                              /*size=*/size,
                              /*weight_ptr=*/weight_ptr,
                              /*adam_v_ptr=*/adam_v_ptr,
                              /*adam_m_ptr=*/adam_m_ptr);
  } else {
    cpu_adam_update_task( 
                         /*alpha_t=*/alpha_t,
                         /*beta1=*/beta1,
                         /*beta2=*/beta2,
                         /*weight_decay=*/weight_decay,
                         /*epsilon=*/epsilon,
                         /*weight_grad_ptr=*/weight_grad_ptr,
                         /*size=*/size,
                         /*num_replicas=*/num_replicas,
                         /*weight_ptr=*/weight_ptr,
                         /*adam_v_ptr=*/adam_v_ptr,
                         /*adam_m_ptr=*/adam_m_ptr);
  }
}

} // namespace FlexFlow
