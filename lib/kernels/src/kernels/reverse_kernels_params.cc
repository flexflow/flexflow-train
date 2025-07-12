#include "kernels/reverse_kernels_params.h"
#include "op-attrs/tensor_dims.h"
#include "utils/nonnegative_int/nonnegative_range.h"

namespace FlexFlow {

ReverseKernelsParams
    compute_reverse_kernels_params(TensorDims const &output_dims,
                                   ReverseAttrs const &attrs) {
  auto axis = attrs.axis;
  positive_int in_blk_size = 1_p;
  positive_int reverse_dim_size = 1_p;
  positive_int num_out_blks = 1_p;
  for (nonnegative_int i : nonnegative_range(get_num_dims(output_dims))) {
    if (i < axis.value) {
      in_blk_size *= dim_at_idx(output_dims, ff_dim_t{i});
    } else if (i == axis.value) {
      reverse_dim_size = dim_at_idx(output_dims, ff_dim_t{i});
    } else {
      num_out_blks *= dim_at_idx(output_dims, ff_dim_t{i});
    }
  }

  return ReverseKernelsParams{
      /*num_out_blks=*/num_out_blks,
      /*reverse_dim_size=*/reverse_dim_size,
      /*in_blk_size=*/in_blk_size,
      /*out_size=*/get_num_elements(output_dims),
  };
}

} // namespace FlexFlow
