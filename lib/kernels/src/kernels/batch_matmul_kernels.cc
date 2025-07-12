#include "kernels/batch_matmul_kernels.h"
#include "kernels/batch_matmul_kernels_cpu.h"
#include "kernels/batch_matmul_kernels_gpu.h"

namespace FlexFlow::Kernels::BatchMatmul {

void forward_kernel(device_stream_t const &stream,
                    device_handle_t const &handle,
                    float *output_ptr,
                    float const *a_input_ptr,
                    float const *b_input_ptr,
                    int m,
                    int n,
                    int k,
                    int batch,
                    int seq_length,
                    int a_seq_length_dim,
                    int b_seq_length_dim) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*handle=*/handle.require_for_gpu(),
        /*output_ptr=*/output_ptr,
        /*a_input_ptr=*/a_input_ptr,
        /*b_input_ptr=*/b_input_ptr,
        /*m=*/m,
        /*n=*/n,
        /*k=*/k,
        /*batch=*/batch,
        /*seq_length=*/seq_length,
        /*a_seq_length_dim=*/a_seq_length_dim,
        /*b_seq_length_dim=*/b_seq_length_dim);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(handle.is_for_cpu());
    cpu_forward_kernel(
        /*output_ptr=*/output_ptr,
        /*a_input_ptr=*/a_input_ptr,
        /*b_input_ptr=*/b_input_ptr,
        /*m=*/m,
        /*n=*/n,
        /*k=*/k,
        /*batch=*/batch,
        /*seq_length=*/seq_length,
        /*a_seq_length_dim=*/a_seq_length_dim,
        /*b_seq_length_dim=*/b_seq_length_dim);
  }
}

void backward_kernel(device_stream_t const &stream,
                     device_handle_t const &handle,
                     float const *o_ptr,
                     float const *o_grad_ptr,
                     float const *a_ptr,
                     float *a_grad_ptr,
                     float const *b_ptr,
                     float *b_grad_ptr,
                     int m,
                     int n,
                     int k,
                     int batch) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*handle=*/handle.require_for_gpu(),
        /*o_ptr=*/o_ptr,
        /*o_grad_ptr=*/o_grad_ptr,
        /*a_ptr=*/a_ptr,
        /*a_grad_ptr=*/a_grad_ptr,
        /*b_ptr=*/b_ptr,
        /*b_grad_ptr=*/b_grad_ptr,
        /*m=*/m,
        /*n=*/n,
        /*k=*/k,
        /*batch=*/batch);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(handle.is_for_cpu());
    cpu_backward_kernel(
        /*o_ptr=*/o_ptr,
        /*o_grad_ptr=*/o_grad_ptr,
        /*a_ptr=*/a_ptr,
        /*a_grad_ptr=*/a_grad_ptr,
        /*b_ptr=*/b_ptr,
        /*b_grad_ptr=*/b_grad_ptr,
        /*m=*/m,
        /*n=*/n,
        /*k=*/k,
        /*batch=*/batch);
  }
}

} // namespace FlexFlow::Kernels::BatchMatmul
