#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ELEMENT_UNARY_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ELEMENT_UNARY_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "kernels/element_unary_per_device_state.dtg.h"
#include "kernels/ff_handle.h"
#include "op-attrs/ops/element_unary_attrs.dtg.h"

namespace FlexFlow::Kernels::ElementUnary {

ElementUnaryPerDeviceState gpu_init_kernel(ArrayShape const &input_shape,
                                           ArrayShape const &output_shape,
                                           ElementUnaryAttrs const &attrs);

void gpu_forward_kernel(ffStream_t stream,
                        ElementUnaryPerDeviceState const &per_device_state,
                        ElementUnaryAttrs const &attrs,
                        PerDeviceFFHandle const &handle,
                        GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output);

void gpu_backward_kernel(ffStream_t stream,
                         ElementUnaryPerDeviceState const &per_device_state,
                         ElementUnaryAttrs const &attrs,
                         PerDeviceFFHandle const &handle,
                         GenericTensorAccessorR const &output,
                         GenericTensorAccessorR const &output_grad,
                         GenericTensorAccessorR const &input,
                         GenericTensorAccessorW const &input_grad);

void gpu_cleanup_kernel(ElementUnaryPerDeviceState &per_device_state);

} // namespace FlexFlow::Kernels::ElementUnary

#endif
