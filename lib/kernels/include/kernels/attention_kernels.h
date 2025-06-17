#ifndef _FLEXFLOW_OPS_KERNELS_ATTENTION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_ATTENTION_KERNELS_H

#include "kernels/allocation.h"
#include "kernels/device.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/mha_per_device_state.h"

namespace FlexFlow::Kernels::MultiHeadAttention {

std::optional<MHAPerDeviceState> init_kernel(
                              DeviceType device_type,
                              PerDeviceFFHandle const &per_device_ff_handle,
                              Allocator &allocator,
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

void forward_kernel(device_stream_t const &stream,
                    std::optional<MHAPerDeviceState> const &device_state,
                    float const *query_ptr,
                    float const *key_ptr,
                    float const *value_ptr,
                    float const *weight_ptr,
                    float *output_ptr);

void backward_kernel(device_stream_t const &stream,
                     std::optional<MHAPerDeviceState> const &device_state,
                     float const *query_ptr,
                     float *query_grad_ptr,
                     float const *key_ptr,
                     float *key_grad_ptr,
                     float const *value_ptr,
                     float *value_grad_ptr,
                     float const *weight_ptr,
                     float *weight_grad_ptr,
                     float const *output_grad_ptr);

void cleanup_kernel(DeviceType device_type,
                    Allocator &allocator,
                    std::optional<MHAPerDeviceState> const &device_state);

} // namespace Kernels::MultiHeadAttention

#endif
