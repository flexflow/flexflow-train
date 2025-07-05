#include "kernels/loss_function_kernels.h"
#include "kernels/loss_function_kernels_cpu.h"
#include "kernels/loss_function_kernels_gpu.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

void sparse_categorical_crossentropy_loss_backward_kernel(
    device_stream_t const &stream,
    float *logit_grad_ptr,
    float const *logit_ptr,
    int const *label_ptr,
    size_t logit_volume,
    size_t logit_grad_volume,
    int num_samples,
    int num_classes,
    int k,
    float scale_factor) {
  if (stream.is_gpu()) {
    sparse_categorical_crossentropy_loss_backward_gpu_kernel(
        /*stream=*/stream.require_gpu(),
        /*logit_grad_ptr=*/logit_grad_ptr,
        /*logit_ptr=*/logit_ptr,
        /*label_ptr=*/label_ptr,
        /*logit_volume=*/logit_volume,
        /*logit_grad_volume=*/logit_grad_volume,
        /*num_samples=*/num_samples,
        /*num_classes=*/num_classes,
        /*k=*/k,
        /*scale_factor=*/scale_factor);
  } else {
    ASSERT(stream.is_cpu());
    sparse_categorical_crossentropy_loss_backward_cpu_kernel(
        /*logit_grad_ptr=*/logit_grad_ptr,
        /*logit_ptr=*/logit_ptr,
        /*label_ptr=*/label_ptr,
        /*logit_volume=*/logit_volume,
        /*logit_grad_volume=*/logit_grad_volume,
        /*num_samples=*/num_samples,
        /*num_classes=*/num_classes,
        /*k=*/k,
        /*scale_factor=*/scale_factor);
  }
}

void categorical_crossentropy_loss_backward_kernel(
    device_stream_t const &stream,
    GenericTensorAccessorW const &logit_grad,
    GenericTensorAccessorR const &logit,
    GenericTensorAccessorR const &label,
    float scale_factor) {
  if (stream.is_gpu()) {
    categorical_crossentropy_loss_backward_gpu_kernel(
        /*stream=*/stream.require_gpu(),
        /*logit_grad_ptr=*/logit_grad.get_float_ptr(),
        /*logit_ptr=*/logit.get_float_ptr(),
        /*label_ptr=*/label.get_float_ptr(),
        /*logit_volume=*/get_num_elements(logit.shape.dims).int_from_positive_int(),
        /*logit_grad_volume=*/
        get_num_elements(logit_grad.shape.dims).int_from_positive_int(),
        /*scale_factor=*/scale_factor);
  } else {
    ASSERT(stream.is_cpu());
    categorical_crossentropy_loss_backward_cpu_kernel(
        /*logit_grad=*/logit_grad,
        /*logit=*/logit,
        /*label=*/label,
        /*scale_factor=*/scale_factor);
  }
}

void mean_squared_error_avg_loss_backward_kernel(device_stream_t const &stream,
                                                 float *logit_grad_ptr,
                                                 float const *logit_ptr,
                                                 float const *label_ptr,
                                                 size_t logit_volume,
                                                 size_t logit_grad_volume,
                                                 float scale_factor) {
  if (stream.is_gpu()) {
    mean_squared_error_avg_loss_backward_gpu_kernel(
        /*stream=*/stream.require_gpu(),
        /*logit_grad_ptr=*/logit_grad_ptr,
        /*logit_ptr=*/logit_ptr,
        /*label_ptr=*/label_ptr,
        /*logit_volume=*/logit_volume,
        /*logit_grad_volume=*/logit_grad_volume,
        /*scale_factor=*/scale_factor);
  } else {
    ASSERT(stream.is_cpu());
    mean_squared_error_avg_loss_backward_cpu_kernel(
        /*logit_grad_ptr=*/logit_grad_ptr,
        /*logit_ptr=*/logit_ptr,
        /*label_ptr=*/label_ptr,
        /*logit_volume=*/logit_volume,
        /*logit_grad_volume=*/logit_grad_volume,
        /*scale_factor=*/scale_factor);
  }
}

void identity_loss_backward_kernel(device_stream_t const &stream,
                                   float *loss_grad_ptr,
                                   float const *loss_ptr,
                                   size_t loss_volume,
                                   size_t loss_grad_volume,
                                   float csale_factor) {
  if (stream.is_gpu()) {
    identity_loss_backward_gpu_kernel(
        /*stream=*/stream.require_gpu(),
        /*loss_grad_ptr=*/loss_grad_ptr,
        /*loss_ptr=*/loss_ptr,
        /*loss_volume=*/loss_volume,
        /*loss_grad_volume=*/loss_grad_volume,
        /*csale_factor=*/csale_factor);
  } else {
    ASSERT(stream.is_cpu());
    identity_loss_backward_cpu_kernel(
        /*loss_grad_ptr=*/loss_grad_ptr,
        /*loss_ptr=*/loss_ptr,
        /*loss_volume=*/loss_volume,
        /*loss_grad_volume=*/loss_grad_volume,
        /*csale_factor=*/csale_factor);
  }
}

} // namespace FlexFlow
