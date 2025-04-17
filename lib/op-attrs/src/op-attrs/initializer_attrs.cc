#include "op-attrs/initializer_attrs.h"
#include "op-attrs/tensor_dims.h"

namespace FlexFlow {

InitializerAttrs make_zero_initializer() {
  return InitializerAttrs{ZeroInitializerAttrs{}};
}

// fan_in and fan_out calculation from pytorch
// see
// https://github.com/pytorch/pytorch/blob/bd019c0bb485904a99fb38589444b1461ab1e486/torch/nn/init.py#L345-L363
static nonnegative_int calculate_fan_for_mode(TensorDims const &dims,
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

// from pytorch:
// see
// https://github.com/pytorch/pytorch/blob/bd019c0bb485904a99fb38589444b1461ab1e486/torch/nn/init.py#L72-L139
static float
    gain_for_nonlinearity(KaimingInitializerNonlinearity nonlinearity,
                          std::optional<float> negative_slope = std::nullopt) {
  if (nonlinearity == KaimingInitializerNonlinearity::RELU) {
    assert(!negative_slope.has_value());
    return sqrtf(2.0);
  } else {
    assert(nonlinearity == KaimingInitializerNonlinearity::LEAKY_RELU);

    return sqrtf(2.0 / (1 + negative_slope.value() * negative_slope.value()));
  }
}

// from pytorch:
// see
// https://github.com/pytorch/pytorch/blob/bd019c0bb485904a99fb38589444b1461ab1e486/torch/nn/init.py#L456-L518
InitializerAttrs kaiming_uniform(TensorDims const &dims,
                                 float a,
                                 KaimingInitializerMode mode,
                                 KaimingInitializerNonlinearity nonlinearity,
                                 int seed) {

  nonnegative_int fan = calculate_fan_for_mode(dims, mode);
  float gain = gain_for_nonlinearity(nonlinearity, a);
  float std = gain / sqrtf(static_cast<float>(fan.unwrap_nonnegative()));
  float bound = sqrtf(3.0) * std;

  return InitializerAttrs{UniformInitializerAttrs{
      /*seed=*/seed,
      /*min_val=*/-bound,
      /*max_val=*/bound,
  }};
}

} // namespace FlexFlow
