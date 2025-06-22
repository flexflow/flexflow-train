#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_NORM_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_NORM_KERNELS_GPU_H

#include "kernels/allocation.h"
#include "kernels/batch_norm_per_device_state.dtg.h"
#include "kernels/ff_handle.h"
#include "kernels/device.h"

namespace FlexFlow::Kernels::BatchNorm {

BatchNormPerDeviceState gpu_init_kernel(PerDeviceFFHandle const &handle,
                                    Allocator &allocator,
                                    float *runningMean,
                                    int output_n,
                                    int output_c,
                                    int output_h,
                                    int output_w,
                                    bool relu);

void gpu_forward_kernel(ffStream_t stream,
                    BatchNormPerDeviceState const &per_device_statem,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *scale_ptr,
                    float const *bias_ptr);

void gpu_backward_kernel(ffStream_t stream,
                     BatchNormPerDeviceState const &per_device_state,
                     float const *output_ptr,
                     float *output_grad_ptr,
                     float const *input_ptr,
                     float *input_grad_ptr,
                     float const *scale_ptr,
                     float *scale_grad_ptr,
                     float *bias_grad_ptr,
                     size_t numElements);

void gpu_cleanup_kernel(Allocator &allocator,
                        BatchNormPerDeviceState &per_device_state);

} // namespace FlexFlow

#endif
