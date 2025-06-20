#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_DROPOUT_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_DROPOUT_KERNELS_GPU_H

#include "kernels/allocation.h"
#include "kernels/array_shape.h"
#include "kernels/ff_handle.h"
#include "kernels/dropout_per_device_state.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include <cstddef>

namespace FlexFlow::Kernels::Dropout {

DropoutPerDeviceState 
  gpu_init_kernel(PerDeviceFFHandle const &handle,
                                float rate,
                                unsigned long long seed,
                                ArrayShape const &output_domain,
                                Allocator &allocator);

void gpu_forward_kernel(ffStream_t stream,
                    DropoutPerDeviceState const &per_device_state,
                    float const *input_ptr,
                    float *output_ptr);

void gpu_backward_kernel(ffStream_t stream,
                     DropoutPerDeviceState const &per_device_state,
                     float const *output_grad_ptr,
                     float *input_grad_ptr);

void gpu_cleanup_kernel(Allocator &allocator,
                    DropoutPerDeviceState const &per_device_state);



} // namespace FlexFlow

#endif
