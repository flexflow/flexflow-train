#include "kernels/split_kernels.h"
#include "kernels/split_kernels_cpu.h"
#include "kernels/split_kernels_gpu.h"
#include <libassert/assert.hpp>

namespace FlexFlow::Kernels::Split {

void forward_kernel(device_stream_t const &stream,
                    float **out_ptrs,
                    float const *in_ptr,
                    int const *out_blk_sizes,
                    int in_blk_size,
                    int num_blks,
                    int numOutputs) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*out_ptrs=*/out_ptrs,
        /*in_ptr=*/in_ptr,
        /*out_blk_sizes=*/out_blk_sizes,
        /*in_blk_size=*/in_blk_size,
        /*num_blks=*/num_blks,
        /*numOutputs=*/numOutputs);
  } else {
    cpu_forward_kernel(
        /*out_ptrs=*/out_ptrs,
        /*in_ptr=*/in_ptr,
        /*out_blk_sizes=*/out_blk_sizes,
        /*in_blk_size=*/in_blk_size,
        /*num_blks=*/num_blks,
        /*numOutputs=*/numOutputs);
  }
}

void backward_kernel(device_stream_t const &stream,
                     float *in_grad_ptr,
                     float const **out_grad_ptr,
                     int const *out_blk_sizes,
                     int in_blk_size,
                     int num_blks,
                     int numOutputs) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*in_grad_ptr=*/in_grad_ptr,
        /*out_grad_ptr=*/out_grad_ptr,
        /*out_blk_sizes=*/out_blk_sizes,
        /*in_blk_size=*/in_blk_size,
        /*num_blks=*/num_blks,
        /*numOutputs=*/numOutputs);
  } else {
    ASSERT(stream.is_cpu());
    cpu_backward_kernel(
        /*in_grad_ptr=*/in_grad_ptr,
        /*out_grad_ptr=*/out_grad_ptr,
        /*out_blk_sizes=*/out_blk_sizes,
        /*in_blk_size=*/in_blk_size,
        /*num_blks=*/num_blks,
        /*numOutputs=*/numOutputs);
  }
}

} // namespace FlexFlow::Kernels::Split
