#include "op-attrs/ops/conv_2d/conv_2d_input_shape.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

Conv2DInputShape parse_input_shape(TensorShape const &input) {
  assert(num_dims(input) == 4);

  nonnegative_int num_samples = dim_at_idx(input, relative_ff_dim_t{0});
  nonnegative_int in_channels = dim_at_idx(input, relative_ff_dim_t{1});
  nonnegative_int in_height = dim_at_idx(input, relative_ff_dim_t{2});
  nonnegative_int in_width = dim_at_idx(input, relative_ff_dim_t{3});

  return Conv2DInputShape{
      num_samples,
      in_channels,
      in_height,
      in_width,
      input.data_type,
  };
}

} // namespace FlexFlow
