#ifndef _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H

#include "kernels/device_stream_t.dtg.h"

namespace FlexFlow::Kernels::Split {

void forward_kernel(device_stream_t const &stream,
                    float **out_ptrs,
                    float const *in_ptr,
                    int const *out_blk_sizes,
                    int in_blk_size,
                    int num_blks,
                    int numOutputs);

void backward_kernel(device_stream_t const &stream,
                     float *in_grad_ptr,
                     float const **out_grad_ptr,
                     int const *out_blk_sizes,
                     int in_blk_size,
                     int num_blks,
                     int numOutputs);

} // namespace FlexFlow::Kernels::Split

#endif // _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H
