#ifndef _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H

#include "kernels/ff_handle.h"
#include "kernels/device.h"
#include "op-attrs/datatype.h"
#include "op-attrs/ops/linear_attrs.dtg.h"
#include "kernels/linear_per_device_state.dtg.h"

namespace FlexFlow::Kernels::Linear {

LinearPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                 float *one_ptr,
                                 std::optional<Activation> activation,
                                 std::optional<RegularizerAttrs> regularizer,
                                 bool use_bias,
                                 DataType input_type,
                                 DataType weight_type,
                                 DataType output_type,
                                 int batch_size,
                                 int channel);

bool use_activation(Activation activation);

void forward_kernel(ffStream_t stream,
                    LinearPerDeviceState const &m,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *filter_ptr,
                    float const *bias_ptr,
                    int in_dim,
                    int out_dim,
                    int batch_size);

void backward_kernel(ffStream_t stream,
                     LinearPerDeviceState const &m,
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

} // namespace Kernels::Linear

#endif
