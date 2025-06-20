#ifndef _FLEXFLOW_OPS_KERNELS_CONV_2D_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CONV_2D_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/ff_handle.h"
#include "op-attrs/activation.dtg.h"
#include "kernels/conv_2d_per_device_state.dtg.h"
#include "kernels/device_stream_t.dtg.h"

namespace FlexFlow::Kernels::Conv2D {

std::optional<Conv2DPerDeviceState> init_kernel(DeviceType device_type,
                                                PerDeviceFFHandle handle,
                                 std::optional<Activation> activation,
                                 int kernel_h,
                                 int kernel_w,
                                 int groups,
                                 int padding_h,
                                 int padding_w,
                                 int stride_h,
                                 int stride_w,
                                 GenericTensorAccessorW const &input,
                                 GenericTensorAccessorW const &output,
                                 float const *filter_ptr,
                                 float *filter_grad_ptr);

void forward_kernel(device_stream_t const &stream,
                    std::optional<Conv2DPerDeviceState> const &per_device_state,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *filter_ptr,
                    float const *bias_ptr,
                    std::optional<Activation> activation);

void backward_kernel(device_stream_t const &stream,
                     std::optional<Conv2DPerDeviceState> const &per_device_state,
                     float const *output_ptr,
                     float *output_grad_ptr,
                     float const *input_ptr,
                     float *input_grad_ptr,
                     float const *filter_ptr,
                     float *filter_grad_ptr,
                     float *bias_grad_ptr,
                     std::optional<Activation> activation);

void cleanup_kernel(DeviceType device_type,
                    std::optional<Conv2DPerDeviceState> &per_device_state);

} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_CONV_2D_KERNELS_H
