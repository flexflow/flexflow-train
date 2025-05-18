#include "kernels/reverse_kernels_params.h"

namespace FlexFlow {

ReverseKernelsParams
    compute_reverse_kernels_params(ArrayShape const &output_shape,
                                   ReverseAttrs const &attrs) {
  auto axis = attrs.axis;
  nonnegative_int in_blk_size = 1_n;
  nonnegative_int reverse_dim_size = 1_n;
  nonnegative_int num_out_blks = 1_n;
  for (nonnegative_int i : nonnegative_range(output_shape.get_dim())) {
    if (i < axis.value) {
      in_blk_size *= output_shape.at(ff_dim_t{i});
    } else if (i == axis.value) {
      reverse_dim_size = output_shape.at(ff_dim_t{i});
    } else {
      num_out_blks *= output_shape.at(ff_dim_t{i});
    }
  }

  return ReverseKernelsParams{
      num_out_blks,
      reverse_dim_size,
      in_blk_size,
      output_shape.get_volume(),
  };
}

} // namespace FlexFlow
