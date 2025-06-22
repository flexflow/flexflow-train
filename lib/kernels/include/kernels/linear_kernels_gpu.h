#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LINEAR_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LINEAR_KERNELS_GPU_H

#include "kernels/ff_handle.h"
#include "kernels/device.h"
#include "kernels/linear_per_device_state.dtg.h"
#include "pcg/device_type.dtg.h"

namespace FlexFlow::Kernels::Linear {

LinearPerDeviceState gpu_init_kernel(PerDeviceFFHandle handle,
                                 float *one_ptr,
                                 std::optional<Activation> activation,
                                 std::optional<RegularizerAttrs> regularizer,
                                 bool use_bias,
                                 DataType input_type,
                                 DataType weight_type,
                                 DataType output_type,
                                 int batch_size,
                                 int channel);

void gpu_forward_kernel(ffStream_t stream,
                    LinearPerDeviceState const &per_device_state,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *filter_ptr,
                    float const *bias_ptr,
                    int in_dim,
                    int out_dim,
                    int batch_size);

void gpu_backward_kernel(ffStream_t stream,
                     LinearPerDeviceState const &per_device_state,
                     float const *output_ptr,
                     float *output_grad_ptr,
                     float const *input_ptr,
                     float *input_grad_ptr,
                     float const *kernel_ptr,
                     float *kernel_grad_ptr,
                     float *bias_grad_ptr,
                     int in_dim,
                     int out_dim,
                     int batch_size);

void gpu_cleanup_kernel(LinearPerDeviceState &per_device_state);

} // namespace FlexFlow

#endif
