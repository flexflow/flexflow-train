#include "substitutions/unity_substitution_set.h"
#include "pcg/machine_specification.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/substitution_builder.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/get_only.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/nonnegative_int/nonnegative_range.h"

namespace FlexFlow {

std::vector<Substitution>
    get_substitution_set(MachineSpecification const &resources) {
  std::vector<Substitution> substitutions;
  for (nonnegative_int dim :
       nonnegative_range(1_n, nonnegative_int{MAX_TENSOR_DIM})) {
    for (nonnegative_int degree = 1_n; degree <= get_num_gpus(resources);
         degree *= 2_n) {
      substitutions.push_back(
          create_replicate_linear_combine(dim, degree, true));
      substitutions.push_back(
          create_replicate_linear_combine(dim, degree, false));
      substitutions.push_back(
          create_partition_linear_combine(dim, degree, true));
      substitutions.push_back(
          create_partition_linear_combine(dim, degree, false));
      substitutions.push_back(
          create_partition_relu_combine(ff_dim_t{dim}, degree));
      substitutions.push_back(
          create_partition_add_combine(ff_dim_t{dim}, degree));
      substitutions.push_back(create_partition_attention_combine(dim, degree));
      substitutions.push_back(create_replicate_attention_reduce(dim, degree));
    }
  }
  for (nonnegative_int degree = 1_n; degree <= get_num_gpus(resources);
       degree *= 2_n) {
    substitutions.push_back(create_partition_conv2d_combine(4_n, degree));
  }

  for (nonnegative_int partition_dim :
       nonnegative_range(1_n, nonnegative_int{MAX_TENSOR_DIM})) {
    for (nonnegative_int softmax_dim :
         nonnegative_range(1_n, nonnegative_int{MAX_TENSOR_DIM})) {
      for (nonnegative_int degree = 1_n; degree <= get_num_gpus(resources);
           degree *= 2_n) {
        if (partition_dim != softmax_dim) {
          substitutions.push_back(create_partition_softmax_combine(
              ff_dim_t{partition_dim}, ff_dim_t{softmax_dim}, degree));
        }
      }
    }
  }
  substitutions.push_back(create_fuse_linear_activation(Activation::RELU));
  substitutions.push_back(create_fuse_linear_activation(Activation::SIGMOID));
  substitutions.push_back(create_fuse_linear_activation(Activation::TANH));
  substitutions.push_back(create_fuse_linear_activation(Activation::GELU));
  return substitutions;
}

Substitution create_replicate_linear_combine(nonnegative_int num_dims,
                                             nonnegative_int degree,
                                             bool use_bias) {
  SubstitutionBuilder b;

  auto [p_input, o_input] = b.add_input(tensor_attribute_pattern_match_all());
  auto [p_weight, o_weight] = b.add_input(tensor_attribute_pattern_match_all());
  std::vector<PatternValue> p_inputs = {p_input, p_weight};

  std::optional<OutputGraphExprValue> o_bias = std::nullopt;
  if (use_bias) {
    std::pair<PatternValue, OutputGraphExprValue> bias =
        b.add_input(tensor_attribute_pattern_match_all());
    p_inputs.push_back(bias.first);
    o_bias = bias.second;
  }

  OperatorAttributePattern linear_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::LINEAR),
      op_attr_key_equals(OperatorAttributeKey::BIAS,
                         OperatorAttributeValue{use_bias}),
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS, degree),
  }};

  PatternValue p_linear_output = get_only(
      b.add_pattern_node(linear_pattern,
                         p_inputs,
                         {tensor_attr_pattern_require_num_dims(num_dims)},
                         "linear"));

  OutputOperatorAttrsAssignment replicate_input_expr =
      OutputOperatorAttrsAssignment{
          std::nullopt,
          {
              set_op_type_attr(OperatorType::REPLICATE),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                   OperatorAttributeValue{degree}),
          }};
  OutputGraphExprValue o_replicate_input_output =
      get_only(b.add_output_graph_node(replicate_input_expr, {o_input}, 1_n));

  OutputOperatorAttrsAssignment partition_weights_expr =
      OutputOperatorAttrsAssignment{
          std::nullopt,
          {
              set_op_type_attr(OperatorType::REPARTITION),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                   OperatorAttributeValue{degree}),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DIM,
                                   OperatorAttributeValue{ff_dim_t{1_n}}),
          }};
  OutputGraphExprValue o_partition_weights_output = get_only(
      b.add_output_graph_node(partition_weights_expr, {o_weight}, 1_n));

  std::vector<OutputGraphExprValue> o_linear_inputs = {
      o_replicate_input_output, o_partition_weights_output};

  if (use_bias) {
    OutputOperatorAttrsAssignment partition_bias_expr =
        OutputOperatorAttrsAssignment{
            std::nullopt,
            {
                set_op_type_attr(OperatorType::REPARTITION),
                set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                     OperatorAttributeValue{degree}),
                set_attr_to_constant(OperatorAttributeKey::PARALLEL_DIM,
                                     OperatorAttributeValue{ff_dim_t{1_n}}),
            }};
    OutputGraphExprValue o_partition_bias_output = get_only(
        b.add_output_graph_node(partition_bias_expr, {o_bias.value()}, 1_n));
    o_linear_inputs.push_back(o_partition_bias_output);
  }

  OutputOperatorAttrsAssignment linear_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named("linear"),
      {},
  };
  OutputGraphExprValue o_linear_output =
      get_only(b.add_output_graph_node(linear_expr, o_linear_inputs, 1_n));

  OutputOperatorAttrsAssignment combine_expr = OutputOperatorAttrsAssignment{
      std::nullopt,
      {
          set_op_type_attr(OperatorType::COMBINE),
          set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                               OperatorAttributeValue{degree}),
          set_attr_to_constant(
              OperatorAttributeKey::PARALLEL_DIM,
              OperatorAttributeValue{ff_dim_t{
                  nonnegative_int{num_dims.unwrap_nonnegative() - 1},
              }}),
      },
  };
  OutputGraphExprValue o_combine_output =
      get_only(b.add_output_graph_node(combine_expr, {o_linear_output}, 1_n));

  b.equate_outputs(p_linear_output, o_combine_output);

  return b.get_substitution();
}

Substitution create_partition_linear_combine(nonnegative_int num_dims,
                                             nonnegative_int degree,
                                             bool use_bias) {
  SubstitutionBuilder b;

  auto [p_input, o_input] = b.add_input(tensor_attribute_pattern_match_all());
  auto [p_weight, o_weight] = b.add_input(tensor_attribute_pattern_match_all());
  std::vector<PatternValue> p_inputs = {p_input, p_weight};

  std::optional<OutputGraphExprValue> o_bias = std::nullopt;
  if (use_bias) {
    std::pair<PatternValue, OutputGraphExprValue> bias =
        b.add_input(tensor_attribute_pattern_match_all());
    p_inputs.push_back(bias.first);
    o_bias = bias.second;
  }

  OperatorAttributePattern linear_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::LINEAR),
      op_attr_key_equals(OperatorAttributeKey::BIAS,
                         OperatorAttributeValue{use_bias}),
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS, degree),
  }};

  PatternValue p_linear_output = get_only(
      b.add_pattern_node(linear_pattern,
                         p_inputs,
                         {tensor_attr_pattern_require_num_dims(num_dims)},
                         "linear"));

  OutputOperatorAttrsAssignment partition_input_expr =
      OutputOperatorAttrsAssignment{
          std::nullopt,
          {
              set_op_type_attr(OperatorType::REPARTITION),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                   OperatorAttributeValue{degree}),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DIM,
                                   OperatorAttributeValue{ff_dim_t{0_n}}),
          }};
  OutputGraphExprValue o_partition_input_output =
      get_only(b.add_output_graph_node(partition_input_expr, {o_input}, 1_n));

  OutputOperatorAttrsAssignment replicate_weights_expr =
      OutputOperatorAttrsAssignment{
          std::nullopt,
          {
              set_op_type_attr(OperatorType::REPLICATE),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                   OperatorAttributeValue{degree}),
          }};
  OutputGraphExprValue o_replicate_weights_output = get_only(
      b.add_output_graph_node(replicate_weights_expr, {o_weight}, 1_n));

  std::vector<OutputGraphExprValue> o_linear_inputs = {
      o_partition_input_output, o_replicate_weights_output};

  if (use_bias) {
    OutputOperatorAttrsAssignment replicate_bias_expr =
        OutputOperatorAttrsAssignment{
            std::nullopt,
            {
                set_op_type_attr(OperatorType::REPLICATE),
                set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                     OperatorAttributeValue{degree}),
            }};
    OutputGraphExprValue o_replicate_bias_output = get_only(
        b.add_output_graph_node(replicate_bias_expr, {o_bias.value()}, 1_n));
    o_linear_inputs.push_back(o_replicate_bias_output);
  }

  OutputOperatorAttrsAssignment linear_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named("linear"),
      {},
  };
  OutputGraphExprValue o_linear_output =
      get_only(b.add_output_graph_node(linear_expr, o_linear_inputs, 1_n));

  OutputOperatorAttrsAssignment combine_expr = OutputOperatorAttrsAssignment{
      std::nullopt,
      {
          set_op_type_attr(OperatorType::COMBINE),
          set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                               OperatorAttributeValue{degree}),
          set_attr_to_constant(
              OperatorAttributeKey::PARALLEL_DIM,
              OperatorAttributeValue{ff_dim_t{
                  nonnegative_int{num_dims.unwrap_nonnegative() - 1},
              }}),
      },
  };
  OutputGraphExprValue o_combine_output =
      get_only(b.add_output_graph_node(combine_expr, {o_linear_output}, 1_n));

  b.equate_outputs(p_linear_output, o_combine_output);

  return b.get_substitution();
}

Substitution create_partition_conv2d_combine(nonnegative_int num_dims,
                                             nonnegative_int degree) {
  if (num_dims != 4) {
    throw mk_runtime_error(fmt::format("num_dims must be 4, not {}", num_dims));
  }

  SubstitutionBuilder b;

  auto [p_input, o_input] = b.add_input(tensor_attribute_pattern_match_all());
  auto [p_weight, o_weight] = b.add_input(tensor_attribute_pattern_match_all());
  std::vector<PatternValue> p_inputs = {p_input, p_weight};

  OperatorAttributePattern conv2d_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::CONV2D),
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS, degree),
  }};

  PatternValue p_conv2d_output = get_only(
      b.add_pattern_node(conv2d_pattern,
                         p_inputs,
                         {tensor_attr_pattern_require_num_dims(num_dims)},
                         "conv2d"));

  OutputOperatorAttrsAssignment partition_input_expr =
      OutputOperatorAttrsAssignment{
          std::nullopt,
          {
              set_op_type_attr(OperatorType::REPARTITION),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                   OperatorAttributeValue{degree}),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DIM,
                                   OperatorAttributeValue{ff_dim_t{0_n}}),
          }};

  OutputGraphExprValue o_partition_input_output =
      get_only(b.add_output_graph_node(partition_input_expr, {o_input}, 1_n));

  OutputOperatorAttrsAssignment replicate_weights_expr =
      OutputOperatorAttrsAssignment{
          std::nullopt,
          {
              set_op_type_attr(OperatorType::REPLICATE),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                   OperatorAttributeValue{degree}),
          }};
  OutputGraphExprValue o_replicate_weights_output = get_only(
      b.add_output_graph_node(replicate_weights_expr, {o_weight}, 1_n));

  std::vector<OutputGraphExprValue> o_conv2d_inputs = {
      o_partition_input_output, o_replicate_weights_output};

  OutputOperatorAttrsAssignment conv2d_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named("conv2d"),
      {},
  };
  OutputGraphExprValue o_conv2d_output =
      get_only(b.add_output_graph_node(conv2d_expr, o_conv2d_inputs, 1_n));

  OutputOperatorAttrsAssignment combine_expr = OutputOperatorAttrsAssignment{
      std::nullopt,
      {
          set_op_type_attr(OperatorType::COMBINE),
          set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                               OperatorAttributeValue{degree}),
          set_attr_to_constant(
              OperatorAttributeKey::PARALLEL_DIM,
              OperatorAttributeValue{ff_dim_t{
                  nonnegative_int{num_dims.unwrap_nonnegative() - 1},
              }}),
      },
  };
  OutputGraphExprValue o_combine_output =
      get_only(b.add_output_graph_node(combine_expr, {o_conv2d_output}, 1_n));

  b.equate_outputs(p_conv2d_output, o_combine_output);

  return b.get_substitution();
}

Substitution create_partition_attention_combine(nonnegative_int num_heads,
                                                nonnegative_int degree) {

  SubstitutionBuilder b;

  auto [p_query_input, o_query_input] =
      b.add_input(tensor_attribute_pattern_match_all());
  auto [p_key_input, o_key_input] =
      b.add_input(tensor_attribute_pattern_match_all());
  auto [p_value_input, o_value_input] =
      b.add_input(tensor_attribute_pattern_match_all());
  auto [p_weights, o_weights] =
      b.add_input(tensor_attribute_pattern_match_all());
  std::vector<PatternValue> p_inputs = {
      p_query_input, p_key_input, p_value_input, p_weights};

  OperatorAttributePattern attention_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::MULTIHEAD_ATTENTION),
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS, degree),
      op_attr_key_divisible_by(OperatorAttributeKey::NUM_HEADS, num_heads),
  }};

  PatternValue p_attention_output =
      get_only(b.add_pattern_node(attention_pattern,
                                  p_inputs,
                                  {tensor_attr_pattern_require_num_dims(3_n)},
                                  "attention"));

  OutputOperatorAttrsAssignment partition_input_expr =
      OutputOperatorAttrsAssignment{
          std::nullopt,
          {
              set_op_type_attr(OperatorType::REPARTITION),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                   OperatorAttributeValue{degree}),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DIM,
                                   OperatorAttributeValue{ff_dim_t{0_n}}),
          }};

  OutputGraphExprValue o_partition_query_input_output = get_only(
      b.add_output_graph_node(partition_input_expr, {o_query_input}, 1_n));

  OutputGraphExprValue o_partition_key_input_output = get_only(
      b.add_output_graph_node(partition_input_expr, {o_key_input}, 1_n));

  OutputGraphExprValue o_partition_value_input_output = get_only(
      b.add_output_graph_node(partition_input_expr, {o_value_input}, 1_n));

  OutputOperatorAttrsAssignment replicate_weight_expr =
      OutputOperatorAttrsAssignment{
          std::nullopt,
          {
              set_op_type_attr(OperatorType::REPLICATE),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                   OperatorAttributeValue{degree}),
          }};

  OutputGraphExprValue o_replicate_weight_output = get_only(
      b.add_output_graph_node(replicate_weight_expr, {o_weights}, 1_n));

  std::vector<OutputGraphExprValue> o_attention_inputs = {
      o_partition_query_input_output,
      o_partition_key_input_output,
      o_partition_value_input_output,
      o_replicate_weight_output};

  OutputOperatorAttrsAssignment attention_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named("attention"),
      {},
  };
  OutputGraphExprValue o_attention_output = get_only(
      b.add_output_graph_node(attention_expr, o_attention_inputs, 1_n));

  OutputOperatorAttrsAssignment combine_expr = OutputOperatorAttrsAssignment{
      std::nullopt,
      {
          set_op_type_attr(OperatorType::COMBINE),
          set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                               OperatorAttributeValue{degree}),
          set_attr_to_constant(OperatorAttributeKey::PARALLEL_DIM,
                               OperatorAttributeValue{ff_dim_t{
                                   2_n,
                               }}),
      },
  };
  OutputGraphExprValue o_combine_output = get_only(
      b.add_output_graph_node(combine_expr, {o_attention_output}, 1_n));

  b.equate_outputs(p_attention_output, o_combine_output);

  return b.get_substitution();
}

Substitution create_replicate_attention_reduce(nonnegative_int num_heads,
                                               nonnegative_int degree) {

  SubstitutionBuilder b;

  auto [p_query_input, o_query_input] =
      b.add_input(tensor_attribute_pattern_match_all());
  auto [p_key_input, o_key_input] =
      b.add_input(tensor_attribute_pattern_match_all());
  auto [p_value_input, o_value_input] =
      b.add_input(tensor_attribute_pattern_match_all());
  auto [p_weights, o_weights] =
      b.add_input(tensor_attribute_pattern_match_all());
  std::vector<PatternValue> p_inputs = {
      p_query_input, p_key_input, p_value_input, p_weights};

  OperatorAttributePattern attention_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::MULTIHEAD_ATTENTION),
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS, degree),
      op_attr_key_divisible_by(OperatorAttributeKey::NUM_HEADS, num_heads),
  }};

  PatternValue p_attention_output =
      get_only(b.add_pattern_node(attention_pattern,
                                  p_inputs,
                                  {tensor_attr_pattern_require_num_dims(3_n)},
                                  "attention"));

  OutputOperatorAttrsAssignment replicate_input_expr =
      OutputOperatorAttrsAssignment{
          std::nullopt,
          {
              set_op_type_attr(OperatorType::REPLICATE),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                   OperatorAttributeValue{degree}),
          }};

  OutputGraphExprValue o_replicate_query_input_output = get_only(
      b.add_output_graph_node(replicate_input_expr, {o_query_input}, 1_n));

  OutputGraphExprValue o_replicate_key_input_output = get_only(
      b.add_output_graph_node(replicate_input_expr, {o_key_input}, 1_n));

  OutputGraphExprValue o_replicate_value_input_output = get_only(
      b.add_output_graph_node(replicate_input_expr, {o_value_input}, 1_n));

  OutputOperatorAttrsAssignment partition_weight_expr =
      OutputOperatorAttrsAssignment{
          std::nullopt,
          {
              set_op_type_attr(OperatorType::REPARTITION),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                   OperatorAttributeValue{degree}),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DIM,
                                   OperatorAttributeValue{ff_dim_t{1_n}}),
          }};

  OutputGraphExprValue o_partition_weight_output = get_only(
      b.add_output_graph_node(partition_weight_expr, {o_weights}, 1_n));

  std::vector<OutputGraphExprValue> o_attention_inputs = {
      o_replicate_query_input_output,
      o_replicate_key_input_output,
      o_replicate_value_input_output,
      o_partition_weight_output};

  OutputOperatorAttrsAssignment attention_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named("attention"),
      {},
  };
  OutputGraphExprValue o_attention_output = get_only(
      b.add_output_graph_node(attention_expr, o_attention_inputs, 1_n));

  OutputOperatorAttrsAssignment reduce_expr = OutputOperatorAttrsAssignment{
      std::nullopt,
      {
          set_op_type_attr(OperatorType::REDUCTION),
          set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                               OperatorAttributeValue{degree}),
      },
  };
  OutputGraphExprValue o_reduce_output =
      get_only(b.add_output_graph_node(reduce_expr, {o_attention_output}, 1_n));

  b.equate_outputs(p_attention_output, o_reduce_output);

  return b.get_substitution();
}

Substitution create_partition_softmax_combine(ff_dim_t softmax_dim,
                                              ff_dim_t partition_dim,
                                              nonnegative_int degree) {
  if (partition_dim == softmax_dim) {
    throw mk_runtime_error(
        fmt::format("partition dim {} must not be equal to softmax dim {}",
                    partition_dim,
                    softmax_dim));
  }
  SubstitutionBuilder b;

  auto [p_input, o_input] = b.add_input(tensor_attribute_pattern_match_all());
  std::vector<PatternValue> p_inputs = {p_input};

  OperatorAttributePattern softmax_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::SOFTMAX),
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS, degree),
      op_attr_key_divisible_by(OperatorAttributeKey::SOFTMAX_DIM,
                               softmax_dim.value),
  }};

  PatternValue p_softmax_output =
      get_only(b.add_pattern_node(softmax_pattern,
                                  p_inputs,
                                  {tensor_attribute_pattern_match_all()},
                                  "softmax"));

  OutputOperatorAttrsAssignment partition_input_expr =
      OutputOperatorAttrsAssignment{
          std::nullopt,
          {
              set_op_type_attr(OperatorType::REPARTITION),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                   OperatorAttributeValue{degree}),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DIM,
                                   OperatorAttributeValue{partition_dim}),
          }};

  OutputGraphExprValue o_partition_input_output =
      get_only(b.add_output_graph_node(partition_input_expr, {o_input}, 1_n));

  std::vector<OutputGraphExprValue> o_softmax_inputs = {
      o_partition_input_output};

  OutputOperatorAttrsAssignment softmax_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named("softmax"),
      {},
  };
  OutputGraphExprValue o_softmax_output =
      get_only(b.add_output_graph_node(softmax_expr, o_softmax_inputs, 1_n));

  OutputOperatorAttrsAssignment combine_expr = OutputOperatorAttrsAssignment{
      std::nullopt,
      {
          set_op_type_attr(OperatorType::COMBINE),
          set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                               OperatorAttributeValue{degree}),
          set_attr_to_constant(OperatorAttributeKey::PARALLEL_DIM,
                               OperatorAttributeValue{partition_dim}),
      },
  };
  OutputGraphExprValue o_combine_output =
      get_only(b.add_output_graph_node(combine_expr, {o_softmax_output}, 1_n));

  b.equate_outputs(p_softmax_output, o_combine_output);

  return b.get_substitution();
}

Substitution create_partition_add_combine(ff_dim_t parallel_dim,
                                          nonnegative_int degree) {
  SubstitutionBuilder b;

  auto [p_input1, o_input1] = b.add_input(tensor_attribute_pattern_match_all());
  auto [p_input2, o_input2] = b.add_input(tensor_attribute_pattern_match_all());
  std::vector<PatternValue> p_inputs = {p_input1, p_input2};

  OperatorAttributePattern add_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::EW_ADD),
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS, degree),
  }};

  PatternValue p_add_output = get_only(b.add_pattern_node(
      add_pattern, p_inputs, {tensor_attribute_pattern_match_all()}, "add"));

  OutputOperatorAttrsAssignment partition_input_expr =
      OutputOperatorAttrsAssignment{
          std::nullopt,
          {
              set_op_type_attr(OperatorType::REPARTITION),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                   OperatorAttributeValue{degree}),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DIM,
                                   OperatorAttributeValue{parallel_dim}),
          }};

  OutputGraphExprValue o_partition_input1_output =
      get_only(b.add_output_graph_node(partition_input_expr, {o_input1}, 1_n));

  OutputGraphExprValue o_partition_input2_output =
      get_only(b.add_output_graph_node(partition_input_expr, {o_input2}, 1_n));

  std::vector<OutputGraphExprValue> o_add_inputs = {o_partition_input1_output,
                                                    o_partition_input2_output};

  OutputOperatorAttrsAssignment add_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named("add"),
      {},
  };
  OutputGraphExprValue o_add_output =
      get_only(b.add_output_graph_node(add_expr, o_add_inputs, 1_n));

  OutputOperatorAttrsAssignment combine_expr = OutputOperatorAttrsAssignment{
      std::nullopt,
      {
          set_op_type_attr(OperatorType::COMBINE),
          set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                               OperatorAttributeValue{degree}),
          set_attr_to_constant(OperatorAttributeKey::PARALLEL_DIM,
                               OperatorAttributeValue{parallel_dim}),
      },
  };
  OutputGraphExprValue o_combine_output =
      get_only(b.add_output_graph_node(combine_expr, {o_add_output}, 1_n));

  b.equate_outputs(p_add_output, o_combine_output);

  return b.get_substitution();
}

Substitution create_partition_relu_combine(ff_dim_t parallel_dim,
                                           nonnegative_int degree) {
  SubstitutionBuilder b;

  auto [p_input, o_input] = b.add_input(tensor_attribute_pattern_match_all());

  OperatorAttributePattern relu_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::RELU),
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS, degree),
  }};

  PatternValue p_relu_output = get_only(b.add_pattern_node(
      relu_pattern, {p_input}, {tensor_attribute_pattern_match_all()}, "relu"));

  OutputOperatorAttrsAssignment partition_input_expr =
      OutputOperatorAttrsAssignment{
          std::nullopt,
          {
              set_op_type_attr(OperatorType::REPARTITION),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                   OperatorAttributeValue{degree}),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DIM,
                                   OperatorAttributeValue{parallel_dim}),
          }};

  OutputGraphExprValue o_partition_input_output =
      get_only(b.add_output_graph_node(partition_input_expr, {o_input}, 1_n));

  OutputOperatorAttrsAssignment relu_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named("relu"),
      {},
  };
  OutputGraphExprValue o_relu_output = get_only(
      b.add_output_graph_node(relu_expr, {o_partition_input_output}, 1_n));

  OutputOperatorAttrsAssignment combine_expr = OutputOperatorAttrsAssignment{
      std::nullopt,
      {
          set_op_type_attr(OperatorType::COMBINE),
          set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                               OperatorAttributeValue{degree}),
          set_attr_to_constant(OperatorAttributeKey::PARALLEL_DIM,
                               OperatorAttributeValue{parallel_dim}),
      },
  };
  OutputGraphExprValue o_combine_output =
      get_only(b.add_output_graph_node(combine_expr, {o_relu_output}, 1_n));

  b.equate_outputs(p_relu_output, o_combine_output);

  return b.get_substitution();
}

Substitution create_fuse_linear_activation(Activation activation) {
  SubstitutionBuilder b;

  auto [p_input, o_input] =
      b.add_input(tensor_attribute_pattern_match_all(), "input");
  auto [p_weight, o_weight] =
      b.add_input(tensor_attribute_pattern_match_all(), "weight");

  OperatorAttributePattern mm_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::LINEAR),
      op_attr_key_equals(
          OperatorAttributeKey::ACTIVATION,
          OperatorAttributeValue{std::optional<Activation>{std::nullopt}}),
  }};
  PatternValue p_mm_output =
      get_only(b.add_pattern_node(mm_pattern,
                                  {p_input, p_weight},
                                  {tensor_attribute_pattern_match_all()},
                                  "mm"));

  OperatorAttributePattern relu_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::RELU),
  }};
  PatternValue p_relu_output =
      get_only(b.add_pattern_node(relu_pattern,
                                  {p_mm_output},
                                  {tensor_attribute_pattern_match_all()},
                                  "relu"));

  OutputOperatorAttrsAssignment fused_node_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named("mm"),
      {
          set_attr_to_constant(OperatorAttributeKey::ACTIVATION,
                               OperatorAttributeValue{activation}),
      }};
  OutputGraphExprValue o_fused_node_output = get_only(
      b.add_output_graph_node(fused_node_expr, {o_input, o_weight}, 1_n));

  b.equate_outputs(p_relu_output, o_fused_node_output);

  return b.get_substitution();
}

} // namespace FlexFlow
