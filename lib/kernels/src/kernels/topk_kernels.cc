#include "kernels/topk_kernels.h"
#include "kernels/topk_kernels_cpu.h"
#include "kernels/topk_kernels_gpu.h"
#include <libassert/assert.hpp>

namespace FlexFlow::Kernels::TopK {

void forward_kernel(device_stream_t const &stream,
                    float const *input_ptr,
                    float *output_ptr,
                    int *indices_ptr,
                    size_t batch_size,
                    int length,
                    int k,
                    bool sorted) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*input_ptr=*/input_ptr,
        /*output_ptr=*/output_ptr,
        /*indices_ptr=*/indices_ptr,
        /*batch_size=*/batch_size,
        /*length=*/length,
        /*k=*/k,
        /*sorted=*/sorted);
  } else {
    ASSERT(stream.is_cpu());
    cpu_forward_kernel(
        /*input_ptr=*/input_ptr,
        /*output_ptr=*/output_ptr,
        /*indices_ptr=*/indices_ptr,
        /*batch_size=*/batch_size,
        /*length=*/length,
        /*k=*/k,
        /*sorted=*/sorted);
  }
}

void backward_kernel(device_stream_t const &stream,
                     float const *out_grad_ptr,
                     int const *indices_ptr,
                     float *in_grad_ptr,
                     size_t batch_size,
                     int length,
                     int k) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*out_grad_ptr=*/out_grad_ptr,
        /*indices_ptr=*/indices_ptr,
        /*in_grad_ptr=*/in_grad_ptr,
        /*batch_size=*/batch_size,
        /*length=*/length,
        /*k=*/k);
  } else {
    ASSERT(stream.is_cpu());
    cpu_backward_kernel(
        /*out_grad_ptr=*/out_grad_ptr,
        /*indices_ptr=*/indices_ptr,
        /*in_grad_ptr=*/in_grad_ptr,
        /*batch_size=*/batch_size,
        /*length=*/length,
        /*k=*/k);
  }
}

} // namespace FlexFlow::Kernels::TopK
