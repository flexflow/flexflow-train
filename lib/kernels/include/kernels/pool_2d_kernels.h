#ifndef _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H

#include "kernels/device_stream_t.dtg.h"
#include "kernels/ff_handle.h"
#include "op-attrs/activation.dtg.h"
#include "op-attrs/ops/pool_2d.h"
#include "kernels/pool_2d_per_device_state.dtg.h"
#include "pcg/device_type.dtg.h"

namespace FlexFlow::Kernels::Pool2D {

std::optional<Pool2DPerDeviceState> init_kernel(DeviceType device_type, 
                                            PerDeviceFFHandle handle,
                                 std::optional<Activation> activation,
                                 int input_w,
                                 int input_h,
                                 int input_c,
                                 int input_n,
                                 int output_w,
                                 int output_h,
                                 int output_c,
                                 int output_n,
                                 int pad_h,
                                 int pad_w,
                                 int kernel_h,
                                 int kernel_w,
                                 int stride_h,
                                 int stride_w,
                                 PoolOp pool_type);

void forward_kernel(device_stream_t const &stream,
                    std::optional<Pool2DPerDeviceState> const &per_device_state,
                    void const *input_ptr,
                    void *output_ptr);

void backward_kernel(device_stream_t const &stream,
                     std::optional<Pool2DPerDeviceState> const &per_device_state,
                     void const *output_ptr,
                     void const *output_grad_ptr,
                     void const *input_ptr,
                     void *input_grad_ptr);

void cleanup_kernel(DeviceType device_type, 
                    std::optional<Pool2DPerDeviceState> &per_device_state);

} // namespace Kernels::Pool2D

#endif // _FLEXFLOW_OPS_KERNELS_POOL_2D_KERNELS_H
