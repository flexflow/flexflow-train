#ifndef _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H

#include "kernels/allocation.h"
#include "kernels/array_shape.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/dropout_per_device_state.dtg.h"
#include <cstddef>

namespace FlexFlow::Kernels::Dropout {

std::optional<DropoutPerDeviceState>
    init_kernel(DeviceType device_type,
                device_handle_t const &handle,
                float rate,
                unsigned long long seed,
                ArrayShape const &output_domain,
                Allocator &allocator);

void forward_kernel(
    device_stream_t const &stream,
    std::optional<DropoutPerDeviceState> const &per_device_state,
    float const *input_ptr,
    float *output_ptr);

void backward_kernel(
    device_stream_t const &stream,
    std::optional<DropoutPerDeviceState> const &per_device_state,
    float const *output_grad_ptr,
    float *input_grad_ptr);

void cleanup_kernel(DeviceType device_type,
                    Allocator &allocator,
                    std::optional<DropoutPerDeviceState> &per_device_state);

} // namespace FlexFlow::Kernels::Dropout

#endif // _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H
