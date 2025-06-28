#include "kernels/linear_kernels_cpu.h"
#include "utils/exception.h"
#include "utils/nonnegative_int/nonnegative_range.h"

namespace FlexFlow::Kernels::Linear {

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output,
                        GenericTensorAccessorR const &projection,
                        std::optional<GenericTensorAccessorR> const &bias) {
  ASSERT(input.shape.num_dims() == 2);
  ASSERT(output.shape.num_dims() == 2);
  ASSERT(projection.shape.num_dims() == 2);
  if (bias.has_value()) {
    ASSERT(bias.value().shape.num_dims() == 1);
  }
  ASSERT(input.shape.at(ff_dim_t{1_n}) == projection.shape.at(ff_dim_t{0_n}));

  for (nonnegative_int i : nonnegative_range(input.shape.at(ff_dim_t{0_n}))) {
    for (nonnegative_int j : nonnegative_range(projection.shape.at(ff_dim_t{1_n}))) {
      float accum = 0.0f;
      if (bias.has_value()) {
        accum += bias.value().at<DataType::FLOAT>(FFOrdered{j});
      }
      for (nonnegative_int k : nonnegative_range(input.shape.at(ff_dim_t{1_n}))) {
        accum += input.at<DataType::FLOAT>(FFOrdered{i, k}) * projection.at<DataType::FLOAT>(FFOrdered{k, j});
      }
      output.at<DataType::FLOAT>(FFOrdered{i, j}) = accum;
    }
  }
}

void cpu_backward_kernel(float const *output_ptr,
                         float *output_grad_ptr,
                         float const *input_ptr,
                         float *input_grad_ptr,
                         float const *kernel_ptr,
                         float *kernel_grad_ptr,
                         float *bias_grad_ptr,
                         int in_dim,
                         int out_dim,
                         int batch_size) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::Linear
