#ifndef _FLEXFLOW_OPS_KERNELS_TOPK_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_TOPK_KERNELS_H

#include "kernels/device_stream_t.dtg.h"

namespace FlexFlow::Kernels::TopK {

void forward_kernel(device_stream_t const &stream,
                    float const *input_ptr,
                    float *output_ptr,
                    int *indices_ptr,
                    size_t batch_size,
                    int length,
                    int k,
                    bool sorted);

void backward_kernel(device_stream_t const &stream,
                     float const *out_grad_ptr,
                     int const *indices_ptr,
                     float *in_grad_ptr,
                     size_t batch_size,
                     int length,
                     int k);

} // namespace Kernels::TopK

#endif // _FLEXFLOW_OPS_KERNELS_TOPK_KERNELS_H
