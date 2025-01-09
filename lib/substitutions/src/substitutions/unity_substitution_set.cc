#include "substitutions/unity_substitution_set.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/substitution_builder.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/get_only.h"

namespace FlexFlow {

Substitution create_combine_inception(int num_convs, int num_dims, int degree) {
  NOT_IMPLEMENTED();
}

Substitution create_combine_concat(int num_inputs, int num_dims, int degree) {
  NOT_IMPLEMENTED();
}

Substitution
    create_replicate_linear_combine(int num_dims, int degree, bool use_bias) {
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
              set_attr_to_constant(OperatorAttributeKey::OP_TYPE,
                                   OperatorType::REPLICATE),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                   degree),
          }};
  OutputGraphExprValue o_replicate_input_output =
      get_only(b.add_output_graph_node(replicate_input_expr, {o_input}, 1));

  OutputOperatorAttrsAssignment partition_weights_expr =
      OutputOperatorAttrsAssignment{
          std::nullopt,
          {
              set_attr_to_constant(OperatorAttributeKey::OP_TYPE,
                                   OperatorType::REPARTITION),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                   degree),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DIM,
                                   ff_dim_t{1}),
          }};
  OutputGraphExprValue o_partition_weights_output =
      get_only(b.add_output_graph_node(partition_weights_expr, {o_weight}, 1));

  std::vector<OutputGraphExprValue> o_linear_inputs = {
      o_replicate_input_output, o_partition_weights_output};

  if (use_bias) {
    OutputOperatorAttrsAssignment partition_bias_expr =
        OutputOperatorAttrsAssignment{
            std::nullopt,
            {
                set_attr_to_constant(OperatorAttributeKey::OP_TYPE,
                                     OperatorType::REPARTITION),
                set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                     degree),
                set_attr_to_constant(OperatorAttributeKey::PARALLEL_DIM,
                                     ff_dim_t{1}),
            }};
    OutputGraphExprValue o_partition_bias_output = get_only(
        b.add_output_graph_node(partition_bias_expr, {o_bias.value()}, 1));
    o_linear_inputs.push_back(o_partition_bias_output);
  }

  OutputOperatorAttrsAssignment linear_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named("linear"),
      {},
  };
  OutputGraphExprValue o_linear_output =
      get_only(b.add_output_graph_node(linear_expr, o_linear_inputs, 1));

  OutputOperatorAttrsAssignment combine_expr = OutputOperatorAttrsAssignment{
      std::nullopt,
      {
          set_attr_to_constant(OperatorAttributeKey::OP_TYPE,
                               OperatorType::COMBINE),
          set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE, degree),
          set_attr_to_constant(OperatorAttributeKey::PARALLEL_DIM,
                               ff_dim_t{num_dims - 1}),
      },
  };
  OutputGraphExprValue o_combine_output =
      get_only(b.add_output_graph_node(combine_expr, {o_linear_output}, 1));

  b.equate_outputs(p_linear_output, o_combine_output);

  return b.get_substitution();
}

Substitution create_partition_linear_combine(int num_dims,
                                             int degree,
                                             Activation activation,
                                             bool use_bias) {
  NOT_IMPLEMENTED();
}

Substitution create_partition_linear_combine(int degree,
                                             Activation activation) {
  NOT_IMPLEMENTED();
}

Substitution create_partition_conv2d_combine(int num_dims, int degree) {
  NOT_IMPLEMENTED();
}

Substitution create_partition_attention_combine(int num_heads, int degree) {
  NOT_IMPLEMENTED();
}

Substitution create_replicate_attention_reduce(int num_heads, int degree) {
  NOT_IMPLEMENTED();
}

Substitution create_partition_add_combine(ff_dim_t parallel_dim, int degree) {
  NOT_IMPLEMENTED();
}

Substitution create_partition_relu_combine(ff_dim_t parallel_dim, int degree) {
  NOT_IMPLEMENTED();
}

Substitution create_partition_concat_combine(int num_inputs,
                                             ff_dim_t concat_dim,
                                             ff_dim_t parallel_dim,
                                             int degree) {
  NOT_IMPLEMENTED();
}

Substitution create_partition_softmax_combine(ff_dim_t softmax_dim,
                                              ff_dim_t partition_dim,
                                              int degree) {
  NOT_IMPLEMENTED();
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
      b.add_output_graph_node(fused_node_expr, {o_input, o_weight}, 1));

  b.equate_outputs(p_relu_output, o_fused_node_output);

  return b.get_substitution();
}

} // namespace FlexFlow
