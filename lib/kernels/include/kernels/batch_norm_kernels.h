#ifndef _FLEXFLOW_KERNELS_BATCH_NORM_KERNELS_H
#define _FLEXFLOW_KERNELS_BATCH_NORM_KERNELS_H

#include "kernels/allocation.h"
#include "kernels/batch_norm_per_device_state.dtg.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/ff_handle.h"

namespace FlexFlow::Kernels::BatchNorm {

std::optional<BatchNormPerDeviceState>
    init_kernel(DeviceType device_type,
                device_handle_t const &handle,
                Allocator &allocator,
                float *runningMean,
                int output_n,
                int output_c,
                int output_h,
                int output_w,
                bool relu);

void forward_kernel(device_stream_t const &stream,
                    BatchNormPerDeviceState const &per_device_state,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *scale_ptr,
                    float const *bias_ptr);

void backward_kernel(device_stream_t const &stream,
                     BatchNormPerDeviceState const &per_device_state,
                     float const *output_ptr,
                     float *output_grad_ptr,
                     float const *input_ptr,
                     float *input_grad_ptr,
                     float const *scale_ptr,
                     float *scale_grad_ptr,
                     float *bias_grad_ptr,
                     size_t numElements);

void cleanup_kernel(
    DeviceType device_type,
    Allocator &allocator,
    std::optional<BatchNormPerDeviceState> const &per_device_state);

} // namespace FlexFlow::Kernels::BatchNorm
#endif
