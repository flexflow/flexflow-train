#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ATTENTION_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ATTENTION_KERNELS_CPU_H

#include "kernels/allocation.h"
#include "kernels/device.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/ff_handle.h"
#include <memory>
#include "kernels/mha_per_device_state.h"

namespace FlexFlow::Kernels::MultiHeadAttention {

void cpu_forward_kernel(float const *query_ptr,
                        float const *key_ptr,
                        float const *value_ptr,
                        float const *weight_ptr,
                        float *output_ptr);

void cpu_backward_kernel(float const *query_ptr,
                         float *query_grad_ptr,
                         float const *key_ptr,
                         float *key_grad_ptr,
                         float const *value_ptr,
                         float *value_grad_ptr,
                         float const *weight_ptr,
                         float *weight_grad_ptr,
                         float const *output_grad_ptr);

} 

#endif
