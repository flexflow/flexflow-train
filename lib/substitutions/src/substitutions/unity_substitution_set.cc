#include "substitutions/unity_substitution_set.h"
#include "pcg/machine_compute_specification.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/substitution_builder.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_only_key.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/nonnegative_int/nonnegative_range.h"

namespace FlexFlow {

std::vector<Substitution>
    get_substitution_set(MachineComputeSpecification const &resources) {
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
  std::unordered_map<TensorSlotName, PatternValue> p_inputs = {
      {TensorSlotName::INPUT, p_input},
      {TensorSlotName::WEIGHT, p_weight},
  };

  std::optional<OutputGraphExprValue> o_bias = std::nullopt;
  if (use_bias) {
    std::pair<PatternValue, OutputGraphExprValue> bias =
        b.add_input(tensor_attribute_pattern_match_all());
    p_inputs.insert({
        TensorSlotName::BIAS,
        bias.first,
    });
    o_bias = bias.second;
  }

  OperatorAttributePattern linear_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::LINEAR),
      op_attr_key_equals(OperatorAttributeKey::BIAS,
                         OperatorAttributeValue{use_bias}),
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS,
                               nonnegative_int{degree}),
  }};

  PatternValue p_linear_output = require_only_key(
      b.add_pattern_node(linear_pattern,
                         p_inputs,
                         {
                             {
                                 TensorSlotName::OUTPUT,
                                 tensor_attr_pattern_require_num_dims(
                                     nonnegative_int{num_dims}),
                             },
                         },
                         "linear"),
      TensorSlotName::OUTPUT);

  OutputOperatorAttrsAssignment replicate_input_expr =
      OutputOperatorAttrsAssignment{
          std::nullopt,
          {
              set_op_type_attr(OperatorType::REPLICATE),
              set_attr_to_constant(OperatorAttributeKey::PARALLEL_DEGREE,
                                   OperatorAttributeValue{degree}),
          }};
  OutputGraphExprValue o_replicate_input_output =
      require_only_key(b.add_output_graph_node(
                           /*node_expr=*/replicate_input_expr,
                           /*inputs=*/
                           {
                               {
                                   TensorSlotName::INPUT,
                                   o_input,
                               },
                           },
                           /*output_slots=*/
                           {
                               TensorSlotName::OUTPUT,
                           }),
                       TensorSlotName::OUTPUT);

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
  OutputGraphExprValue o_partition_weights_output =
      require_only_key(b.add_output_graph_node(
                           /*node_expr=*/partition_weights_expr,
                           /*inputs=*/
                           {
                               {
                                   TensorSlotName::INPUT,
                                   o_weight,
                               },
                           },
                           /*output_slots=*/
                           {
                               TensorSlotName::OUTPUT,
                           }),
                       TensorSlotName::OUTPUT);

  std::unordered_map<TensorSlotName, OutputGraphExprValue> o_linear_inputs = {
      {
          TensorSlotName::INPUT,
          o_replicate_input_output,
      },
      {
          TensorSlotName::WEIGHT,
          o_partition_weights_output,
      },
  };

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
    OutputGraphExprValue o_partition_bias_output =
        require_only_key(b.add_output_graph_node(
                             /*node_expr=*/partition_bias_expr,
                             /*inputs=*/
                             {
                                 {
                                     TensorSlotName::INPUT,
                                     o_bias.value(),
                                 },
                             },
                             /*output_slots=*/
                             {
                                 TensorSlotName::OUTPUT,
                             }),
                         TensorSlotName::OUTPUT);
    o_linear_inputs.insert({
        TensorSlotName::BIAS,
        o_partition_bias_output,
    });
  }

  OutputOperatorAttrsAssignment linear_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named("linear"),
      {},
  };
  OutputGraphExprValue o_linear_output =
      require_only_key(b.add_output_graph_node(
                           /*node_expr=*/linear_expr,
                           /*inputs=*/o_linear_inputs,
                           /*output_slots=*/
                           {
                               TensorSlotName::OUTPUT,
                           }),
                       TensorSlotName::OUTPUT);

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
      require_only_key(b.add_output_graph_node(
                           /*node_expr=*/combine_expr,
                           /*inputs=*/
                           {
                               {
                                   TensorSlotName::INPUT,
                                   o_linear_output,
                               },
                           },
                           /*output_slots=*/
                           {
                               TensorSlotName::OUTPUT,
                           }),
                       TensorSlotName::OUTPUT);

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
      require_only_key(b.add_pattern_node(
                           /*node_expr=*/mm_pattern,
                           /*inputs=*/
                           {
                               {
                                   TensorSlotName::INPUT,
                                   p_input,
                               },
                               {
                                   TensorSlotName::WEIGHT,
                                   p_weight,
                               },
                           },
                           /*output_patterns=*/
                           {
                               {
                                   TensorSlotName::OUTPUT,
                                   tensor_attribute_pattern_match_all(),
                               },
                           },
                           /*name=*/"mm"),
                       TensorSlotName::OUTPUT);

  OperatorAttributePattern relu_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::RELU),
  }};
  PatternValue p_relu_output =
      require_only_key(b.add_pattern_node(
                           /*node_expr=*/relu_pattern,
                           /*inputs=*/
                           {
                               {
                                   TensorSlotName::INPUT,
                                   p_mm_output,
                               },
                           },
                           /*output_patterns=*/
                           {
                               {
                                   TensorSlotName::OUTPUT,
                                   tensor_attribute_pattern_match_all(),
                               },
                           },
                           /*name=*/"relu"),
                       TensorSlotName::OUTPUT);

  OutputOperatorAttrsAssignment fused_node_expr = OutputOperatorAttrsAssignment{
      b.pattern_node_named("mm"),
      {
          set_attr_to_constant(OperatorAttributeKey::ACTIVATION,
                               OperatorAttributeValue{activation}),
      }};
  OutputGraphExprValue o_fused_node_output =
      require_only_key(b.add_output_graph_node(
                           /*node_expr=*/fused_node_expr,
                           /*inputs=*/
                           {
                               {
                                   TensorSlotName::INPUT,
                                   o_input,
                               },
                               {
                                   TensorSlotName::WEIGHT,
                                   o_weight,
                               },
                           },
                           /*output_slots=*/
                           {
                               TensorSlotName::OUTPUT,
                           }),
                       TensorSlotName::OUTPUT);

  b.equate_outputs(p_relu_output, o_fused_node_output);

  return b.get_substitution();
}

} // namespace FlexFlow
