#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_OPTIMIZER_KERNELS_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_OPTIMIZER_KERNELS_H

#include "kernels/device_stream_t.dtg.h"
#include "kernels/ff_handle.h"

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
                     float *sgd_v_ptr);

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
                      float *adam_m_ptr);

} // namespace FlexFlow

#endif
