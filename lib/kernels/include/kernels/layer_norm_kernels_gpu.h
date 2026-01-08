#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LAYER_NORM_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LAYER_NORM_KERNELS_GPU_H

#include "kernels/allocation.h"
#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "kernels/layer_norm_per_device_state.dtg.h"

namespace FlexFlow::Kernels::LayerNorm {

// todo: this may have some problem.
LayerNormPerDeviceState gpu_init_kernel(PerDeviceFFHandle const &handle,
                                        Allocator &allocator,
                                        bool elementwise_affine,
                                        int64_t effective_batch_size,
                                        int64_t effective_num_elements,
                                        float eps);

void gpu_forward_kernel(ffStream_t stream,
                        LayerNormPerDeviceState const &per_device_state,
                        GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output,
                        GenericTensorAccessorW const &gamma,
                        GenericTensorAccessorW const &beta);

void gpu_backward_kernel(ffStream_t stream,
                         LayerNormPerDeviceState const &per_device_state,
                         GenericTensorAccessorR const &output_grad,
                         GenericTensorAccessorR const &input,
                         GenericTensorAccessorW const &input_grad,
                         GenericTensorAccessorR const &gamma,
                         GenericTensorAccessorW const &gamma_grad,
                         GenericTensorAccessorW const &beta_grad);

void gpu_cleanup_kernel(LayerNormPerDeviceState const &per_device_state);

} // namespace FlexFlow::Kernels::LayerNorm

#endif
