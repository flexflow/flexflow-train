#ifndef _FLEXFLOW_KERNELS_BATCH_NORM_KERNELS_H
#define _FLEXFLOW_KERNELS_BATCH_NORM_KERNELS_H

#include "device.h"
#include "kernels/allocation.h"
#include "kernels/batch_norm_per_device_state.dtg.h"
#include "kernels/ff_handle.h"
#include <memory>

namespace FlexFlow {
namespace Kernels {
namespace BatchNorm {

BatchNormPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                    Allocator allocator,
                                    float *runningMean,
                                    int output_n,
                                    int output_c,
                                    int output_h,
                                    int output_w,
                                    bool relu);

void forward_kernel(ffStream_t stream,
                    BatchNormPerDeviceState const &per_device_statem,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *scale_ptr,
                    float const *bias_ptr);

void backward_kernel(ffStream_t stream,
                     BatchNormPerDeviceState const &per_device_state,
                     float const *input_ptr,
                     float *output_grad_ptr,
                     float const *output_ptr,
                     float *input_grad_ptr,
                     float const *scale_ptr,
                     float *scale_grad_ptr,
                     float *bias_grad_ptr,
                     size_t numElements);

void cleanup_kernel(Allocator allocator,
                    ffTensorDescriptor_t inputTensor,
                    ffTensorDescriptor_t biasTensor,
                    ffTensorDescriptor_t outputTensor,
                    ffActivationDescriptor_t actiDesc,
                    bool relu,
                    float *runningMean);

} // namespace BatchNorm
} // namespace Kernels
} // namespace FlexFlow

#endif
