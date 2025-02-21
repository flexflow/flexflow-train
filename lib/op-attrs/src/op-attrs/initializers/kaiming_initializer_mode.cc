#include "op-attrs/initializers/kaiming_initializer_mode.h"
#include "op-attrs/tensor_dims.h"

namespace FlexFlow {

nonnegative_int calculate_fan_for_mode(TensorDims const &dims,
                                       KaimingInitializerMode mode) {
  nonnegative_int num_input_fmaps = dim_at_idx(dims, relative_ff_dim_t{0});
  nonnegative_int num_output_fmaps = dim_at_idx(dims, relative_ff_dim_t{1});

  nonnegative_int receptive_field_size = get_num_elements(
      slice_tensor_dims(dims, relative_ff_dim_t{2}, std::nullopt));

  if (mode == KaimingInitializerMode::FAN_IN) {
    return num_input_fmaps * receptive_field_size;
  } else {
    assert(mode == KaimingInitializerMode::FAN_OUT);

    return num_output_fmaps * receptive_field_size;
  }
}

} // namespace FlexFlow
