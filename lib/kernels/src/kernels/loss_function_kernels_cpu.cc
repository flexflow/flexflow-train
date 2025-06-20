#include "kernels/loss_function_kernels_cpu.h"
#include "utils/exception.h"

namespace FlexFlow {

void sparse_categorical_crossentropy_loss_backward_cpu_kernel(
    float *logit_grad_ptr,
    float const *logit_ptr,
    int const *label_ptr,
    size_t logit_volume,
    size_t logit_grad_volume,
    int num_samples,
    int num_classes,
    int k,
    float scale_factor) {
  NOT_IMPLEMENTED();
}

void categorical_crossentropy_loss_backward_cpu_kernel(float *logit_grad_ptr,
                                                   float const *logit_ptr,
                                                   float const *label_ptr,
                                                   size_t logit_volume,
                                                   size_t logit_grad_volume,
                                                   float scale_factor) {
  NOT_IMPLEMENTED();
}

void mean_squared_error_avg_loss_backward_cpu_kernel(float *logit_grad_ptr,
                                                 float const *logit_ptr,
                                                 float const *label_ptr,
                                                 size_t logit_volume,
                                                 size_t logit_grad_volume,
                                                 float scale_factor) {
  NOT_IMPLEMENTED();
}

void identity_loss_backward_cpu_kernel(float *loss_grad_ptr,
                                   float const *loss_ptr,
                                   size_t loss_volume,
                                   size_t loss_grad_volume,
                                   float csale_factor) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
