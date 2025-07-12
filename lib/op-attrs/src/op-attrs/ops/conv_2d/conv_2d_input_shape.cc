#include "op-attrs/ops/conv_2d/conv_2d_input_shape.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

Conv2DInputShape parse_input_shape(TensorShape const &input) {
  ASSERT(get_num_dims(input.dims) == 4);

  positive_int num_samples = dim_at_idx(input.dims, relative_ff_dim_t{0});
  positive_int in_channels = dim_at_idx(input.dims, relative_ff_dim_t{1});
  positive_int in_height = dim_at_idx(input.dims, relative_ff_dim_t{2});
  positive_int in_width = dim_at_idx(input.dims, relative_ff_dim_t{3});

  return Conv2DInputShape{
      /*num_samples=*/num_samples,
      /*num_channels=*/in_channels,
      /*height=*/in_height,
      /*width=*/in_width,
      /*datatype=*/input.data_type,
  };
}

} // namespace FlexFlow
