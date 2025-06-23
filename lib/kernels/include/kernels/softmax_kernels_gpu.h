#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SOFTMAX_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SOFTMAX_KERNELS_GPU_H

#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "kernels/legion_dim_t.dtg.h"
#include "kernels/softmax_per_device_state.dtg.h"

namespace FlexFlow::Kernels::Softmax {

SoftmaxPerDeviceState gpu_init_kernel(PerDeviceFFHandle const &handle,
                                      legion_dim_t dim,
                                      int input_n,
                                      int input_c,
                                      int input_h,
                                      int input_w);

void gpu_forward_kernel(ffStream_t stream,
                        SoftmaxPerDeviceState const &per_device_state,
                        float const *input_ptr,
                        float *output_ptr);

void gpu_backward_kernel(ffStream_t stream,
                         float const *output_grad_ptr,
                         float *input_grad_ptr,
                         size_t num_elements);

void gpu_cleanup_kernel(SoftmaxPerDeviceState &per_device_state);

} // namespace FlexFlow::Kernels::Softmax

#endif
