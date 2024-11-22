#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_OPTIMIZER_KERNELS_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_OPTIMIZER_KERNELS_H

#include "device.h"
#include "kernels/ff_handle.h"
#include "kernels/nccl.h"
#include "kernels/per_device_op_state.dtg.h"

namespace FlexFlow {

__global__ void sgd_update(size_t count,
                           float lr,
                           float weight_decay,
                           float momentum,
                           bool nesterov,
                           float const *WGrad,
                           float *V,
                           float *W);

class SGDOptimizer {
public:
  static __host__ void ps_update_task_gpu(SGDOptimizer const *op,
                                          float const *w_grad_ptr,
                                          size_t size,
                                          int num_replicas,
                                          float *w_ptr,
                                          float *v_ptr);

#ifdef FF_USE_NCCL
  static __host__ void nccl_update_task_gpu(SGDOptimizer const *op,
                                            PerDeviceOpState const *meta,
                                            float const *w_grad_ptr,
                                            size_t size,
                                            float *w_ptr,
                                            float *v_ptr);
#endif

public:
  float lr;
  float weight_decay;
  float momentum;
  bool nesterov;
};

__global__ void
    add_kernel(int count, float scale, float const *src, float *dst);

__global__ void scale_kernel(int count, float a, float b, float *ptr);

__global__ void adam_update(int count,
                            float alpha_t,
                            float beta1,
                            float beta2,
                            float weight_decay,
                            float epsilon,
                            float const *WGrad,
                            float *M,
                            float *V,
                            float *W);

class AdamOptimizer {
public:
  static __host__ void ps_update_task_gpu(AdamOptimizer const *op,
                                          float const *w_grad_ptr,
                                          size_t size,
                                          int num_replicas,
                                          float *w_ptr,
                                          float *v_ptr,
                                          float *m_ptr);

#ifdef FF_USE_NCCL
  static __host__ void nccl_update_task_gpu(AdamOptimizer const *op,
                                            PerDeviceOpState const *meta,
                                            float const *w_grad_ptr,
                                            size_t size,
                                            float *w_ptr,
                                            float *v_ptr,
                                            float *m_ptr);
#endif

public:
  float alpha;
  float alpha_t;
  float beta1;
  float beta2;
  float weight_decay;
  float epsilon;
};

} // namespace FlexFlow

#endif // _FLEXFLOW_KERNELS_INCLUDE_KERNELS_OPTIMIZER_KERNELS_H
