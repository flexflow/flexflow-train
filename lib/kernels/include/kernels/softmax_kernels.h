#ifndef _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H

#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/ff_handle.h"
#include "kernels/softmax_per_device_state.dtg.h"
#include "pcg/device_type.dtg.h"

namespace FlexFlow::Kernels::Softmax {

std::optional<SoftmaxPerDeviceState> init_kernel(DeviceType device_type,
                                                 device_handle_t const &handle,
                                                 ff_dim_t dim,
                                                 int input_n,
                                                 int input_c,
                                                 int input_h,
                                                 int input_w);

void forward_kernel(
    device_stream_t const &stream,
    std::optional<SoftmaxPerDeviceState> const &per_device_state,
    float const *input_ptr,
    float *output_ptr);

void backward_kernel(device_stream_t const &stream,
                     float const *output_grad_ptr,
                     float *input_grad_ptr,
                     size_t num_elements);

void cleanup_kernel(DeviceType device_type,
                    std::optional<SoftmaxPerDeviceState> &per_device_state);

} // namespace FlexFlow::Kernels::Softmax

#endif
