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
  for (positive_int dim = 1_p; dim <= positive_int{MAX_TENSOR_DIM}; dim++) {
    for (positive_int degree = 1_p; degree <= get_num_gpus(resources);
         degree *= 2_p) {
      substitutions.push_back(
          create_replicate_linear_combine(dim, degree, true));
      substitutions.push_back(
          create_replicate_linear_combine(dim, degree, false));
      substitutions.push_back(
          create_partition_linear_combine(dim, degree, true));
      substitutions.push_back(
          create_partition_linear_combine(dim, degree, false));
      substitutions.push_back(create_partition_relu_combine(
          ff_dim_t{dim.nonnegative_int_from_positive_int()}, degree));
      substitutions.push_back(create_partition_add_combine(
          ff_dim_t{dim.nonnegative_int_from_positive_int()}, degree));
      substitutions.push_back(create_partition_attention_combine(dim, degree));
      substitutions.push_back(create_replicate_attention_reduce(dim, degree));
    }
  }
  for (positive_int degree = 1_p; degree <= get_num_gpus(resources);
       degree *= 2_p) {
    substitutions.push_back(create_partition_conv2d_combine(4_p, degree));
  }

  for (positive_int partition_dim = 1_p;
       partition_dim <= positive_int{MAX_TENSOR_DIM};
       partition_dim++) {
    for (positive_int softmax_dim = 1_p;
         softmax_dim <= positive_int{MAX_TENSOR_DIM};
         softmax_dim++) {
      for (positive_int degree = 1_p; degree <= get_num_gpus(resources);
           degree *= 2_p) {
        if (partition_dim != softmax_dim) {
          substitutions.push_back(create_partition_softmax_combine(
              ff_dim_t{partition_dim.nonnegative_int_from_positive_int()},
              ff_dim_t{softmax_dim.nonnegative_int_from_positive_int()},
              degree));
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

Substitution create_replicate_linear_combine(positive_int num_dims,
                                             positive_int degree,
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
      op_attr_key_divisible_by(OperatorAttributeKey::OUT_CHANNELS, degree),
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
                  nonnegative_int{num_dims.int_from_positive_int() - 1},
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

Substitution create_partition_linear_combine(positive_int num_dims,
                                             positive_int degree,
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
                  nonnegative_int{num_dims.int_from_positive_int() - 1},
              }}),
      },
  };
  OutputGraphExprValue o_combine_output =
      get_only(b.add_output_graph_node(combine_expr, {o_linear_output}, 1_n));

  b.equate_outputs(p_linear_output, o_combine_output);

  return b.get_substitution();
}

Substitution create_partition_conv2d_combine(positive_int num_dims,
                                             positive_int degree) {
  if (num_dims != 4_p) {
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
                  nonnegative_int{num_dims.int_from_positive_int() - 1},
              }}),
      },
  };
  OutputGraphExprValue o_combine_output =
      get_only(b.add_output_graph_node(combine_expr, {o_conv2d_output}, 1_n));

  b.equate_outputs(p_conv2d_output, o_combine_output);

  return b.get_substitution();
}

Substitution create_partition_attention_combine(positive_int num_heads,
                                                positive_int degree) {

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
                                  {tensor_attr_pattern_require_num_dims(3_p)},
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

Substitution create_replicate_attention_reduce(positive_int num_heads,
                                               positive_int degree) {

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
                                  {tensor_attr_pattern_require_num_dims(3_p)},
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
                                              positive_int degree) {
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
                               positive_int{softmax_dim.value}),
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
                                          positive_int degree) {
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
                                           positive_int degree) {
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
