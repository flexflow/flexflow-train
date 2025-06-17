#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_NORM_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_NORM_KERNELS_CPU_H

#include "kernels/allocation.h"
#include "kernels/batch_norm_per_device_state.dtg.h"
#include "kernels/device_stream_t.dtg.h"

namespace FlexFlow::Kernels::BatchNorm {

void cpu_forward_kernel(BatchNormPerDeviceState const &per_device_state,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *scale_ptr,
                    float const *bias_ptr);

void cpu_backward_kernel(BatchNormPerDeviceState const &per_device_state,
                     float const *output_ptr,
                     float *output_grad_ptr,
                     float const *input_ptr,
                     float *input_grad_ptr,
                     float const *scale_ptr,
                     float *scale_grad_ptr,
                     float *bias_grad_ptr,
                     size_t numElements);

} // namespace FlexFlow

#endif
