#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ATTENTION_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ATTENTION_KERNELS_GPU_H

#include "kernels/allocation.h"
#include "kernels/device.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/ff_handle.h"
#include "kernels/mha_per_device_state.dtg.h"

namespace FlexFlow::Kernels::MultiHeadAttention {

MHAPerDeviceState gpu_init_kernel(PerDeviceFFHandle const &,
                              Allocator &,
                              int num_samples,
                              int num_heads,
                              int qSize,
                              int kSize,
                              int vSize,
                              int qProjSize,
                              int kProjSize,
                              int vProjSize,
                              int oProjSize,
                              int qoSeqLength,
                              int kvSeqLength,
                              bool add_bias_kv);

void gpu_forward_kernel(ffStream_t stream,
                    MHAPerDeviceState const &device_state,
                    float const *query_ptr,
                    float const *key_ptr,
                    float const *value_ptr,
                    float const *weight_ptr,
                    float *output_ptr);

void gpu_backward_kernel(ffStream_t stream,
                     MHAPerDeviceState const &device_state,
                     float const *query_ptr,
                     float *query_grad_ptr,
                     float const *key_ptr,
                     float *key_grad_ptr,
                     float const *value_ptr,
                     float *value_grad_ptr,
                     float const *weight_ptr,
                     float *weight_grad_ptr,
                     float const *output_grad_ptr);

void gpu_cleanup_kernel(Allocator &allocator,
                    MHAPerDeviceState const &device_state);

} // namespace Kernels::MultiHeadAttention

#endif
