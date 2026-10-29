#include "op-attrs/ops/linear.h"
#include "op-attrs/ff_ordered/ff_ordered_slice.h"
#include "op-attrs/ff_ordered/ff_ordered_transform.h"
#include "op-attrs/initializers/kaiming_initializer_mode.h"
#include "op-attrs/num_ptensor_shard_dims_t.h"
#include "op-attrs/num_tensor_dims_t.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/relative_ff_dim_t.h"
#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/product.h"
#include "utils/containers/set_of.h"
#include "utils/expected.h"
#include "utils/fmt/optional.h"
#include "utils/integer_conversions.h"
#include "utils/optional.h"
#include "utils/orthotope/bounded_component.h"
#include "utils/orthotope/dim_projection.h"
#include "utils/orthotope/down_projection.h"
#include "utils/orthotope/eq_projection.h"
#include "utils/orthotope/minimal_dim_domain_mapping.h"
#include "utils/orthotope/orthotope_bounded_coord.h"
#include "utils/orthotope/up_projection.h"

namespace FlexFlow {

std::map<TensorSlotName, IncomingTensorRole>
    get_linear_incoming_tensor_roles(LinearAttrs const &attrs) {
  std::map<TensorSlotName, IncomingTensorRole> result = {
      {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
      {TensorSlotName::WEIGHT, IncomingTensorRole::WEIGHT},
  };

  if (attrs.use_bias) {
    result[TensorSlotName::BIAS] = IncomingTensorRole::WEIGHT;
  }

  return result;
}

std::set<TensorSlotName> linear_get_slots(LinearAttrs const &attrs) {
  std::set<TensorSlotName> result = {
      TensorSlotName::INPUT,
      TensorSlotName::WEIGHT,
      TensorSlotName::OUTPUT,
  };

  if (attrs.use_bias) {
    result.insert(TensorSlotName::BIAS);
  }

  return result;
}

std::map<TensorSlotName, TensorRole>
    linear_get_tensor_roles(LinearAttrs const &attrs) {
  std::map<TensorSlotName, TensorRole> result = {
      {TensorSlotName::INPUT, TensorRole::INPUT},
      {TensorSlotName::WEIGHT, TensorRole::WEIGHT},
      {TensorSlotName::OUTPUT, TensorRole::OUTPUT},
  };

  if (attrs.use_bias) {
    result[TensorSlotName::BIAS] = TensorRole::WEIGHT;
  }

  return result;
}

TensorShape linear_get_projection_shape(LinearAttrs const &attrs,
                                        TensorShape const &input_shape) {
  positive_int in_channels =
      dim_at_idx(input_shape.dims, relative_ff_dim_t{-1});

  return TensorShape{
      TensorDims{
          FFOrdered<positive_int>{attrs.out_channels, in_channels},
      },
      input_shape.data_type,
  };
}

TensorShape linear_get_bias_shape(LinearAttrs const &attrs,
                                  TensorShape const &input_shape) {
  return TensorShape{
      TensorDims{
          FFOrdered<positive_int>{attrs.out_channels},
      },
      input_shape.data_type,
  };
}

TensorShape linear_get_output_shape(LinearAttrs const &attrs,
                                    TensorShape const &input_shape) {
  TensorShape output_shape = input_shape;
  output_shape.dims.ff_ordered.at(relative_ff_dim_t{-1}) = attrs.out_channels;

  return output_shape;
}

std::map<TensorSlotName, TensorShape>
    linear_get_weight_shapes(LinearAttrs const &attrs,
                             TensorShape const &input_shape) {

  std::map<TensorSlotName, TensorShape> weight_shapes = {
      {
          TensorSlotName::WEIGHT,
          linear_get_projection_shape(attrs, input_shape),
      },
  };

  if (attrs.use_bias) {
    weight_shapes.insert({
        TensorSlotName::BIAS,
        linear_get_bias_shape(attrs, input_shape),
    });
  }

  return weight_shapes;
}

//! [parallel shape inference composition example]
ParallelTensorShape
    linear_get_projection_parallel_shape(LinearAttrs const &attrs,
                                         ParallelTensorShape const &input) {
  TensorShape unpar =
      linear_get_projection_shape(attrs, get_reduced_shape(input));

  ParallelTensorDimDegrees projection_degrees =
      linear_get_projection_parallel_dim_degrees(attrs,
                                                 get_parallel_degrees(input));

  return lift_shape_to_parallel_with_degrees(unpar, projection_degrees);
}
//! [parallel shape inference composition example]

ParallelTensorShape
    linear_get_bias_parallel_shape(LinearAttrs const &attrs,
                                   ParallelTensorShape const &input) {

  TensorShape unpar = linear_get_bias_shape(attrs, get_reduced_shape(input));

  ParallelTensorDimDegrees bias_degrees =
      linear_get_bias_parallel_dim_degrees(attrs, get_parallel_degrees(input));

  return lift_shape_to_parallel_with_degrees(unpar, bias_degrees);
}

ParallelTensorShape
    linear_get_output_parallel_shape(LinearAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  TensorShape unpar = linear_get_output_shape(attrs, get_reduced_shape(input));

  ParallelTensorDimDegrees output_degrees =
      linear_get_output_parallel_dim_degrees(attrs,
                                             get_parallel_degrees(input));

  return lift_shape_to_parallel_with_degrees(unpar, output_degrees);
}

ParallelTensorDimDegrees linear_get_projection_parallel_dim_degrees(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input) {
  SumDegree sum_degree = SumDegree{1_p};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      input.sum_degree.value *
      product(ff_ordered_slice(
          input.shard_degrees, relative_ff_dim_t{0}, relative_ff_dim_t{-1}))};
  FFOrdered<positive_int> shard_degrees = FFOrdered<positive_int>{
      input.discard_copy_degree.value,
      input.shard_degrees.at(relative_ff_dim_t{-1}),
  };

  return ParallelTensorDimDegrees{
      /*sum_degree=*/sum_degree,
      /*discard_copy_degree=*/discard_copy_degree,
      /*shard_degrees=*/shard_degrees,
  };
}

ParallelTensorDimDegrees linear_get_bias_parallel_dim_degrees(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input) {

  SumDegree sum_degree = SumDegree{
      input.sum_degree.value * input.shard_degrees.at(relative_ff_dim_t{-1}),
  };
  DiscardCopyDegree discard_copy_degree =
      DiscardCopyDegree{product(ff_ordered_slice(
          input.shard_degrees, relative_ff_dim_t{0}, relative_ff_dim_t{-1}))};
  FFOrdered<positive_int> shard_degrees =
      FFOrdered<positive_int>{input.discard_copy_degree.value};

  return ParallelTensorDimDegrees{
      /*sum_degree=*/sum_degree,
      /*discard_copy_degree=*/discard_copy_degree,
      /*shard_degrees=*/shard_degrees,
  };
}

ParallelTensorDimDegrees linear_get_output_parallel_dim_degrees(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input) {
  SumDegree sum_degree = SumDegree{
      input.sum_degree.value * input.shard_degrees.at(relative_ff_dim_t{-1}),
  };

  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{1_p};
  FFOrdered<positive_int> shard_degrees = input.shard_degrees;
  shard_degrees.at(relative_ff_dim_t{-1}) = input.discard_copy_degree.value;

  return ParallelTensorDimDegrees{
      /*sum_degree=*/sum_degree,
      /*discard_copy_degree=*/discard_copy_degree,
      /*shard_degrees=*/shard_degrees,
  };
}

std::map<TensorSlotName, ParallelTensorShape>
    linear_get_weight_parallel_shapes(LinearAttrs const &attrs,
                                      ParallelTensorShape const &input_shape) {

  std::map<TensorSlotName, ParallelTensorShape> weight_shapes = {
      {
          TensorSlotName::WEIGHT,
          linear_get_projection_parallel_shape(attrs, input_shape),
      },
  };

  if (attrs.use_bias) {
    weight_shapes.insert({TensorSlotName::BIAS,
                          linear_get_bias_parallel_shape(attrs, input_shape)});
  }

  return weight_shapes;
}

/**
 * @brief Chosen to match pytorch implementation
 *
 * see
 * https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/nn/modules/linear.py#L114-L122
 */
std::map<TensorSlotName, InitializerAttrs> linear_get_initializers(
    LinearAttrs const &attrs,
    TensorShape const &input_shape,
    std::optional<InitializerAttrs> const &maybe_projection_initializer,
    std::optional<InitializerAttrs> const &maybe_bias_initializer) {

  ASSERT(attrs.use_bias || !maybe_bias_initializer.has_value(),
         fmt::format("Expected bias_initializer=std::nullopt since "
                     "use_bias=false, but received bias_initializer: {}",
                     maybe_bias_initializer.value()));

  TensorShape projection_shape =
      linear_get_projection_shape(attrs, input_shape);

  InitializerAttrs projection_default_initializer =
      InitializerAttrs{KaimingNormalAttrs{
          /*a=*/sqrtf(5.0),
          /*mode=*/KaimingInitializerMode::FAN_IN,
          /*nonlinearity=*/KaimingInitializerNonlinearity::LEAKY_RELU,
          /*seed=*/0,
      }};

  InitializerAttrs projection_initializer =
      maybe_projection_initializer.value_or(projection_default_initializer);

  positive_int fan_in = calculate_fan_for_mode(projection_shape.dims,
                                               KaimingInitializerMode::FAN_IN);

  float bound = 1 / sqrtf(static_cast<float>(fan_in.int_from_positive_int()));

  InitializerAttrs bias_default_initializer =
      InitializerAttrs{UniformInitializerAttrs{
          /*seed=*/0,
          /*min_val=*/-bound,
          /*max_val=*/bound,
      }};

  InitializerAttrs bias_initializer =
      maybe_bias_initializer.value_or(bias_default_initializer);

  if (attrs.use_bias) {
    return std::map<TensorSlotName, InitializerAttrs>{
        {TensorSlotName::WEIGHT, projection_initializer},
        {TensorSlotName::BIAS, bias_initializer},
    };
  } else {
    return std::map<TensorSlotName, InitializerAttrs>{
        {TensorSlotName::WEIGHT, projection_initializer},
    };
  }
}

StandardOperatorTaskGroup
    linear_get_task_group(LinearAttrs const &attrs,
                          ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees =
      linear_get_output_parallel_dim_degrees(attrs, input_degrees);

  StandardOperatorTaskGroup task_group = StandardOperatorTaskGroup{
      transform(
          get_parallel_tensor_space_coordinates(input_degrees),
          [&](ParallelTensorSpaceCoordinate const &input_coord)
              -> AbstractedOperatorAtomicTaskShardBinding {
            num_ptensor_shard_dims_t input_num_shard_dims =
                get_ptensor_dim_degrees_num_shard_dims(input_degrees);

            parallel_tensor_dim_idx_t input_sum_dim = sum_dim_idx();
            parallel_tensor_dim_idx_t input_discard_copy_dim =
                discard_copy_dim_idx();

            std::set<parallel_tensor_dim_idx_t> input_leading_dims =
                shard_dim_idxs_for_exclusive_interval(
                    0, -1, input_num_shard_dims);

            parallel_tensor_dim_idx_t input_channel_dim =
                shard_dim_idx_for_relative(-1, input_num_shard_dims);

            OrthotopeBoundedCoord data_parallelism_component =
                orthotope_bounded_coord_for_ptensor_dims(
                    input_degrees, input_coord, input_leading_dims);

            BoundedComponent output_channel_parallelism_component =
                bounded_component_for_ptensor_dim(
                    input_degrees, input_coord, input_discard_copy_dim);

            BoundedComponent preexisting_sum_parallelism_component =
                bounded_component_for_ptensor_dim(
                    input_degrees, input_coord, input_sum_dim);

            BoundedComponent new_sum_parallelism_component =
                bounded_component_for_ptensor_dim(
                    input_degrees, input_coord, input_channel_dim);

            BoundedComponent output_sum_component = assert_unwrap(
                flatten_orthotope_bounded_coord(make_2d_orthotope_bounded_coord(
                    preexisting_sum_parallelism_component,
                    new_sum_parallelism_component)));

            BoundedComponent output_discard_copy_component =
                trivial_bounded_component();

            OrthotopeBoundedCoord output_shard_components =
                orthotope_bounded_coord_product(
                    data_parallelism_component,
                    lift_bounded_component(
                        output_channel_parallelism_component));

            OrthotopeBoundedCoord raw_output_coord =
                orthotope_bounded_coord_product(
                    lift_bounded_component(output_sum_component),
                    lift_bounded_component(output_discard_copy_component),
                    output_shard_components);

            ParallelTensorSpaceCoordinate output_coord =
                parallel_tensor_space_coordinate_from_bounded_orthotope_components(
                    /*sum_degree=*/output_sum_component,
                    /*discard_copy_degree=*/output_discard_copy_component,
                    /*shard_coords=*/output_shard_components);

            return AbstractedOperatorAtomicTaskShardBinding{
                /*tensor_corods=*/{
                    {
                        TensorSlotName::INPUT,
                        input_coord,
                    },
                    {
                        TensorSlotName::WEIGHT,
                        parallel_tensor_space_coordinate_from_bounded_orthotope_components(
                            /*sum_coord=*/trivial_bounded_component(),
                            /*discard_copy_coord=*/
                            assert_unwrap(flatten_orthotope_bounded_coord(
                                make_orthotope_bounded_coord_from_head_and_tail(
                                    preexisting_sum_parallelism_component,
                                    data_parallelism_component))),
                            /*shard_coords=*/
                            make_2d_orthotope_bounded_coord(
                                output_channel_parallelism_component,
                                new_sum_parallelism_component)),
                    },
                    {
                        TensorSlotName::BIAS,
                        parallel_tensor_space_coordinate_from_bounded_orthotope_components(
                            /*sum_coord=*/output_sum_component,
                            /*discard_copy_coord=*/
                            assert_unwrap(flatten_orthotope_bounded_coord(
                                data_parallelism_component)),
                            /*shard_coords=*/
                            lift_bounded_component(
                                output_channel_parallelism_component)),
                    },
                    {
                        TensorSlotName::OUTPUT,
                        output_coord,
                    },
                },
                /*task_coord=*/
                task_coord_matching_parallel_tensor_space_coordinate(
                    output_coord, output_degrees),
            };
          }),
  };

  return restrict_standard_operator_task_group_to_slots(
      task_group, linear_get_slots(attrs));
}

OperatorTaskSpace linear_get_operator_task_space(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {

  StandardOperatorTaskGroup op_task_group =
      linear_get_task_group(attrs, input_degrees);

  return task_space_for_standard_operator_task_group(op_task_group);
}

ShardSignatureInstance linear_get_shard_signature_instance(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {

  StandardOperatorTaskGroup op_task_group =
      linear_get_task_group(attrs, input_degrees);

  return shard_signature_instance_from_standard_operator_task_group(
      op_task_group);
}

OperatorSpaceToParallelTensorSpaceMapping
    linear_get_operator_to_projection_mapping(
        LinearAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  StandardOperatorTaskGroup op_task_group =
      linear_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(
      op_task_group, TensorSlotName::WEIGHT);
}

OperatorSpaceToParallelTensorSpaceMapping linear_get_operator_to_input_mapping(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {
  StandardOperatorTaskGroup op_task_group =
      linear_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(
      op_task_group, TensorSlotName::INPUT);
}

OperatorSpaceToParallelTensorSpaceMapping linear_get_operator_to_bias_mapping(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {

  StandardOperatorTaskGroup op_task_group =
      linear_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(
      op_task_group, TensorSlotName::BIAS);
}

OperatorSpaceToParallelTensorSpaceMapping linear_get_operator_to_output_mapping(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {

  StandardOperatorTaskGroup op_task_group =
      linear_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(
      op_task_group, TensorSlotName::OUTPUT);
}

} // namespace FlexFlow
