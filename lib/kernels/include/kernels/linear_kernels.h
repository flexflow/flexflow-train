#ifndef _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H

#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/ff_handle.h"
#include "kernels/linear_per_device_state.dtg.h"
#include "op-attrs/datatype.h"
#include "op-attrs/ops/linear_attrs.dtg.h"
#include "pcg/device_type.dtg.h"

namespace FlexFlow::Kernels::Linear {

std::optional<LinearPerDeviceState>
    init_kernel(DeviceType device_type,
                device_handle_t const &handle,
                float *one_ptr,
                std::optional<Activation> activation,
                std::optional<RegularizerAttrs> regularizer,
                bool use_bias,
                DataType input_type,
                DataType weight_type,
                DataType output_type,
                int batch_size,
                int channel);

void forward_kernel(device_stream_t const &stream,
                    std::optional<LinearPerDeviceState> const &per_device_state,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *filter_ptr,
                    float const *bias_ptr,
                    int in_dim,
                    int out_dim,
                    int batch_size);

void backward_kernel(
    device_stream_t const &stream,
    std::optional<LinearPerDeviceState> const &per_device_state,
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

void cleanup_kernel(DeviceType device_type,
                    std::optional<LinearPerDeviceState> &per_device_state);

} // namespace FlexFlow::Kernels::Linear

#endif
