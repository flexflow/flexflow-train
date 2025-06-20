#include "kernels/attention_kernels.h"
#include "kernels/attention_kernels_cpu.h"
#include "kernels/attention_kernels_gpu.h"

namespace FlexFlow::Kernels::MultiHeadAttention {

std::optional<MHAPerDeviceState> init_kernel(DeviceType device_type,
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
                              bool add_bias_kv) {
  if (device_type == DeviceType::GPU) {
    return gpu_init_kernel(  
      /*per_device_ff_handle=*/per_device_ff_handle,
      /*allocator=*/allocator,
      /*num_samples=*/num_samples,
      /*num_heads=*/num_heads,
      /*qSize=*/qSize,
      /*kSize=*/kSize,
      /*vSize=*/vSize,
      /*qProjSize=*/qProjSize,
      /*kProjSize=*/kProjSize,
      /*vProjSize=*/vProjSize,
      /*oProjSize=*/oProjSize,
      /*qoSeqLength=*/qoSeqLength,
      /*kvSeqLength=*/kvSeqLength,
      /*add_bias_kv=*/add_bias_kv);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    return std::nullopt;
  }
}

void forward_kernel(device_stream_t const &stream,
                    std::optional<MHAPerDeviceState> const &device_state,
                    float const *query_ptr,
                    float const *key_ptr,
                    float const *value_ptr,
                    float const *weight_ptr,
                    float *output_ptr) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
      /*stream=*/stream.require_gpu(),
      /*device_state=*/device_state.value(),
      /*query_ptr=*/query_ptr,
      /*key_ptr=*/key_ptr,
      /*value_ptr=*/value_ptr,
      /*weight_ptr=*/weight_ptr,
      /*output_ptr=*/output_ptr
    );
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(device_state == std::nullopt);
    cpu_forward_kernel(
      /*query_ptr=*/query_ptr,
      /*key_ptr=*/key_ptr,
      /*value_ptr=*/value_ptr,
      /*weight_ptr=*/weight_ptr,
      /*output_ptr=*/output_ptr
    );
  }

}

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
                     float const *output_grad_ptr) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
      /*stream=*/stream.require_gpu(),
      /*device_state=*/device_state.value(),
      /*query_ptr=*/query_ptr,
      /*query_grad_ptr=*/query_grad_ptr,
      /*key_ptr=*/key_ptr,
      /*key_grad_ptr=*/key_grad_ptr,
      /*value_ptr=*/value_ptr,
      /*value_grad_ptr=*/value_grad_ptr,
      /*weight_ptr=*/weight_ptr,
      /*weight_grad_ptr=*/weight_grad_ptr,
      /*output_grad_ptr=*/output_grad_ptr
    );
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(device_state == std::nullopt);
    cpu_backward_kernel(
      /*query_ptr=*/query_ptr,
      /*query_grad_ptr=*/query_grad_ptr,
      /*key_ptr=*/key_ptr,
      /*key_grad_ptr=*/key_grad_ptr,
      /*value_ptr=*/value_ptr,
      /*value_grad_ptr=*/value_grad_ptr,
      /*weight_ptr=*/weight_ptr,
      /*weight_grad_ptr=*/weight_grad_ptr,
      /*output_grad_ptr=*/output_grad_ptr
    );
  }
}

void cleanup_kernel(DeviceType device_type,
                    Allocator &allocator,
                    std::optional<MHAPerDeviceState> const &device_state) {
  if (device_type == DeviceType::GPU) {
    gpu_cleanup_kernel(allocator, device_state.value());
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(device_state == std::nullopt);
  }
}

} // namespace FlexFlow
