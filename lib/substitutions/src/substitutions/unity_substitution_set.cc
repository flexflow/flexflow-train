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
  for (nonnegative_int num_dims :
       nonnegative_range(1_n, nonnegative_int{MAX_TENSOR_DIM})) {
    for (nonnegative_int degree = 1_n; degree <= get_num_gpus(resources);
         degree *= 2_n) {
      substitutions.push_back(
          create_replicate_linear_combine(num_dims, degree, true));
      substitutions.push_back(
          create_replicate_linear_combine(num_dims, degree, false));
    }
  }
  substitutions.push_back(create_fuse_linear_activation(Activation::RELU));
  substitutions.push_back(create_fuse_linear_activation(Activation::SIGMOID));
  substitutions.push_back(create_fuse_linear_activation(Activation::TANH));
  substitutions.push_back(create_fuse_linear_activation(Activation::GELU));
  return substitutions;
}

Substitution create_combine_inception(nonnegative_int num_convs,
                                      nonnegative_int num_dims,
                                      nonnegative_int degree) {
  NOT_IMPLEMENTED();
}

Substitution create_combine_concat(nonnegative_int num_inputs,
                                   nonnegative_int num_dims,
                                   nonnegative_int degree) {
  NOT_IMPLEMENTED();
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
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS,
                               nonnegative_int{degree}),
  }};

  PatternValue p_linear_output = get_only(b.add_pattern_node(
      linear_pattern,
      p_inputs,
      {tensor_attr_pattern_require_num_dims(nonnegative_int{num_dims})},
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
                                             Activation activation,
                                             bool use_bias) {
  NOT_IMPLEMENTED();
}

Substitution create_partition_conv2d_combine(nonnegative_int num_dims,
                                             nonnegative_int degree) {
  NOT_IMPLEMENTED();
}

Substitution create_partition_attention_combine(nonnegative_int num_heads,
                                                nonnegative_int degree) {
  NOT_IMPLEMENTED();
}

Substitution create_replicate_attention_reduce(nonnegative_int num_heads,
                                               nonnegative_int degree) {
  NOT_IMPLEMENTED();
}

Substitution create_partition_add_combine(ff_dim_t parallel_dim,
                                          nonnegative_int degree) {
  NOT_IMPLEMENTED();
}

Substitution create_partition_relu_combine(ff_dim_t parallel_dim,
                                           nonnegative_int degree) {
  NOT_IMPLEMENTED();
}

Substitution create_partition_concat_combine(nonnegative_int num_inputs,
                                             ff_dim_t concat_dim,
                                             ff_dim_t parallel_dim,
                                             nonnegative_int degree) {
  NOT_IMPLEMENTED();
}

Substitution create_partition_softmax_combine(ff_dim_t softmax_dim,
                                              ff_dim_t partition_dim,
                                              nonnegative_int degree) {
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
      b.add_output_graph_node(fused_node_expr, {o_input, o_weight}, 1_n));

  b.equate_outputs(p_relu_output, o_fused_node_output);

  return b.get_substitution();
}

} // namespace FlexFlow
